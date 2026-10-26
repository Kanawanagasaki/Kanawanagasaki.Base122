using System.Buffers.Binary;
using System.Text;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>Shared test helpers: hex conversion, a deterministic PRNG and stream wrappers.</summary>
public static class TestHelpers
{
    /// <summary>Seed used for randomized-but-reproducible stream chunking tests.</summary>
    public static ulong Seed = 0x1234_5678_9ABC_DEF0UL;
    public static string ToHex(ReadOnlySpan<byte> bytes)
    {
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes)
            sb.Append(b.ToString("x2"));
        return sb.ToString();
    }

    public static byte[] FromHex(string hex)
    {
        if (hex.Length % 2 != 0)
            throw new ArgumentException("Hex string must have an even length.");
        var bytes = new byte[hex.Length / 2];
        for (int i = 0; i < bytes.Length; i++)
            bytes[i] = byte.Parse(hex.Substring(i * 2, 2), System.Globalization.NumberStyles.HexNumber);
        return bytes;
    }

    public static string EncodedVectorToText(string encodedHex) =>
        Encoding.UTF8.GetString(FromHex(encodedHex));

    /// <summary>Deterministic splitmix64 PRNG, matching the reference Rust test suite.</summary>
    public static IEnumerable<byte> RandomBytes(ulong seed, int count)
    {
        for (int i = 0; i < count; i++)
            yield return (byte)(NextU64(ref seed) >> 33);
    }

    public static ulong NextU64(ref ulong seed)
    {
        seed += 0x9E3779B97F4A7C15UL;
        ulong z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static byte[] RandomBytesArray(ulong seed, int count) => RandomBytes(seed, count).ToArray();

    /// <summary>The six ASCII values that Base122 must never emit literally.</summary>
    public static readonly byte[] IllegalBytes = [0, 10, 13, 34, 38, 92];
}

/// <summary>A read-only stream wrapper that hides seeking, to exercise buffered decoding.</summary>
public sealed class NonSeekableStream(Stream inner) : Stream
{
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
    public override int Read(Span<byte> buffer) => inner.Read(buffer);
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

/// <summary>A stream whose <see cref="Read"/> returns at most <paramref name="maxChunk"/> bytes per call.</summary>
public sealed class TrickleStream(Stream inner, int maxChunk) : Stream
{
    public override bool CanRead => true;
    public override bool CanSeek => inner.CanSeek;
    public override bool CanWrite => false;
    public override long Length => inner.Length;
    public override long Position { get => inner.Position; set => inner.Position = value; }

    public override int Read(byte[] buffer, int offset, int count)
    {
        return inner.Read(buffer, offset, Math.Min(count, maxChunk));
    }

    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
    public override void SetLength(long value) => inner.SetLength(value);
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
