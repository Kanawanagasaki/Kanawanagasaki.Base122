# Kanawanagasaki.Base122

**VIBE CODED**

[![NuGet package](https://img.shields.io/nuget/v/Kanawanagasaki.Base122.svg)](https://www.nuget.org/packages/Kanawanagasaki.Base122)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance [Base122](https://blog.kevinalbs.com/base122) binary-to-text encoding for .NET - a C# port of the scalar pipeline of the Rust [base122-fast](https://crates.io/crates/base122-fast) crate.

Base122 is a UTF-8-safe binary-to-text encoding with only **~14% size overhead** over raw bytes (Base64 adds 33%). A 7-bit group maps to one ASCII character; the six characters that are problematic in HTML and URLs (NUL, LF, CR, `"`, `&`, `\`) never appear literally - each is folded, together with the following 7-bit group, into a single two-byte UTF-8 escape character. The output of every encoding API is valid UTF-8 text.

## Features

- **~14% overhead** (Base64 adds 33%), valid UTF-8 output.
- **In-memory API**: `byte[]`/`ReadOnlySpan<byte>` → `string`, and `ReadOnlySpan<char>`/`string` → `byte[]`.
- **UTF-8 byte-form API**: `EncodeToUtf8` / `DecodeFromUtf8` operate directly on the encoded UTF-8 bytes - byte in, byte out, no string or char data is ever materialized.
- **Stream API**: `Base122Encoder` and `Base122Decoder` are themselves `Stream`s (write-only / read-only) - pipe into and out of them with `CopyTo`/`CopyToAsync`, wrap them with `StreamReader`, or compose them with further stream adapters (LZ4, GZip, …). Fully async (`WriteAsync`/`ReadAsync`/`CompleteAsync` with `CancellationToken`).
- **Allocation-conscious**: exact-size `string.Create` for small arrays, `stackalloc` for small spans, `ArrayPool` for large ones, and zero steady-state allocations for the span and stream APIs.
- **Scalar SWAR** with hardware-accelerated bit gather/scatter via BMI2 `PDEP`/`PEXT` (`System.Runtime.Intrinsics`) and a portable fallback.
- **Validated** against the reference Rust implementation: 7,700+ cross-implementation vectors, exhaustive two-byte round-trips, randomized fuzzing and corrupted-input robustness tests.

## Installation

```sh
dotnet add package Kanawanagasaki.Base122
```

## Usage

### In-memory encoding and decoding

```csharp
using Kanawanagasaki.Base122;

byte[] data = File.ReadAllBytes("image.png");

// Encode: byte array (or ReadOnlySpan<byte>) to string.
string encoded = Base122.Encode(data);

// Decode: string (or ReadOnlySpan<char>) to byte array.
byte[] decoded = Base122.Decode(encoded);

Console.WriteLine($"Original: {data.Length} bytes, encoded: {encoded.Length} chars");
```

### Span-based, allocation-free variants

```csharp
// Encode into a caller-provided buffer. The buffer must be at least
// GetMaxEncodedLength(data.Length) chars long.
if (Base122.TryEncode(data, charBuffer, out int charsWritten))
    Console.WriteLine($"{charsWritten} chars written");

// Decode into a caller-provided buffer. The buffer must be at least
// GetMaxDecodedLength(encoded.Length) bytes long.
if (Base122.TryDecode(encoded, byteBuffer, out int bytesWritten))
    Console.WriteLine($"{bytesWritten} bytes written");

// Size helpers.
int exactChars = Base122.GetEncodedLength(data);   // exact encoded length
int maxChars   = Base122.GetMaxEncodedLength(data.Length);
int maxBytes   = Base122.GetMaxDecodedLength(encoded.Length);
```

### UTF-8

`EncodeToUtf8` and `DecodeFromUtf8` produce and consume the encoded text as raw UTF-8 bytes.

```csharp
// Encode: byte array (or ReadOnlySpan<byte>) to the encoded UTF-8 bytes.
byte[] encodedBytes = Base122.EncodeToUtf8(data);

// Decode: encoded UTF-8 bytes back to the original data.
byte[] decodedBytes = Base122.DecodeFromUtf8(encodedBytes);

// Exact O(1) size helpers for the byte form.
int encodedSize   = Base122.GetEncodedToUtf8Length(data);       // exact
int maxEncoded    = Base122.GetMaxEncodedToUtf8Length(data.Length);
int maxDecoded    = Base122.GetMaxDecodedFromUtf8Length(encodedBytes.Length);
int decodedSize   = Base122.GetDecodedFromUtf8Length(encodedBytes); // exact for canonical input

// Caller-provided buffers, zero allocations:
Base122.TryEncodeToUtf8(data, byteBuffer, out int bytesWritten);
Base122.TryDecodeFromUtf8(encodedBytes, dataBuffer, out int bytesDecoded);
```

Decoding invalid input throws `FormatException` (the Try variants return
`false` instead). Both directions allocate exactly one right-sized array per
call; `TryEncodeToUtf8` requires a buffer of at least
`GetMaxEncodedToUtf8Length` bytes - the conservative maximum, which may exceed
the exact encoded length by one byte when escapes occur.

### Streaming: encoder and decoder are Streams

`Base122Encoder` is a write-only `Stream`: raw bytes written to it are encoded into the output stream supplied at construction. `Base122Decoder` is a read-only `Stream`: it reads encoded text (UTF-8) from its input stream and exposes the decoded bytes. Both work with arbitrarily sized chunks - bit-level state (accumulators, pending escapes) is carried across calls transparently.

Because they are `Stream`s, they compose with anything that speaks streams:

```csharp
// Pipe a raw file through the encoder into a compressed stream:
await using var encoder = new Base122Encoder(File.Create("data.b122"));
await rawStream.CopyToAsync(encoder);       // raw -> encoded
// (then e.g. wrap encoder's output in GZip/LZ4 further down)

// Decode by piping out of the decoder:
await using var decoder = new Base122Decoder(File.OpenRead("data.b122"));
await decoder.CopyToAsync(destination);     // encoded -> raw

// Even through a text reader:
using var reader = new StreamReader(decoder);
string text = reader.ReadToEnd();
```

Without piping, the explicit shape is the same:

```csharp
using var output = File.Create("data.b122");
using (var encoder = new Base122Encoder(output))
{
    foreach (ReadOnlySpan<byte> chunk in ReadChunks(rawStream))
        encoder.Write(chunk);

    encoder.Complete(); // emits the final group and trailing escape, then flushes
}
```

Use `leaveOpen: true` to keep the underlying stream open after disposal, `Flush()`/`FlushAsync()` to push buffered output without finalizing, and `Complete()`/`CompleteAsync()` to finalize the bit stream (both are also called automatically on `Dispose`/`DisposeAsync`).

Malformed input throws `InvalidDataException` - an invalid lead byte, an invalid continuation byte, or an escape sequence truncated by the end of the stream.

## How it works

Input is treated as a bit stream, consumed 7 bits at a time:

- A legal group is emitted as a one-byte UTF-8 character (`0xxxxxxx`).
- A group that maps to one of the six illegal characters is emitted as a single two-byte UTF-8 character (`110sss1x 10xxxxxx`) that also carries the next 7-bit group - 14 bits per two-byte character.
- The final group is zero-padded; if it would be illegal, the "shortened" escape (`sss = 111`) marks the ending, and decoding rounds down to whole bytes to recover the exact original length.

## Performance

Single-threaded, .NET 10, linux-x64, Intel i7-9700K, 10 MB payloads, best of 5 (Release build):

| Payload | Encode | Decode | EncodeToUtf8 | DecodeFromUtf8 |
| :--- | ---: | ---: | ---: | ---: |
| random bytes | 268 MB/s | 322 MB/s | 422 MB/s | 349 MB/s |
| zeros (escape-dense) | 359 MB/s | 391 MB/s | 450 MB/s | 303 MB/s |
| illegal-heavy | 361 MB/s | 306 MB/s | 356 MB/s | 287 MB/s |
| ASCII text | 669 MB/s | 350 MB/s | 768 MB/s | 766 MB/s |

Streaming (`Base122Encoder`/`Base122Decoder` with 8 KB chunks): encode 202–294 MB/s, decode 198–552 MB/s.

`Encode(byte[])` allocates exactly the result string, and `EncodeToUtf8`/`DecodeFromUtf8` allocate exactly the result byte array; `TryEncode`/`TryDecode`, `TryEncodeToUtf8`/`TryDecodeFromUtf8` and the streaming classes perform no allocations in steady state. For comparison, the reference Rust crate reaches ~0.9 GB/s with its scalar SWAR codec and ~2–3.5 GB/s with its AVX2 kernels on the same hardware - the C# port currently implements the scalar pipeline (plus BMI2 acceleration), so expect it in that range.

## Notes

- The static API is thread-safe; `Base122Encoder` and `Base122Decoder` instances are not.
- Encoded output is always valid UTF-8 and contains none of `0x00`, `\n`, `\r`, `"`, `&`, `\`.

## License

[MIT](LICENSE) - derived from the Rust [base122-fast](https://crates.io/crates/base122-fast) implementation, which is also MIT licensed.
