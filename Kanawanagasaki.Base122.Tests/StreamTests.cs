using System.Text;
using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>Tests for the streaming <see cref="Base122Encoder"/>.</summary>
public class Base122EncoderStreamTests
{
    [Fact]
    public void Write_Empty_ProducesNoOutput()
    {
        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output);
        encoder.Complete();
        Assert.Empty(output.ToArray());
    }

    [Fact]
    public void Write_MatchesStaticEncoding_ForVariousChunkSizes()
    {
        foreach (int len in new[] { 1, 2, 3, 7, 8, 9, 14, 15, 27, 28, 29, 55, 56, 57, 100, 255, 256, 1000 })
        {
            byte[] data = TestHelpers.RandomBytesArray((ulong)len, len);
            string expected = Base122.Encode(data);

            foreach (int chunkSize in new[] { 1, 2, 3, 5, 7, 8, 13, 64, 4096 })
            {
                using var output = new MemoryStream();
                using (var encoder = new Base122Encoder(output))
                {
                    for (int offset = 0; offset < len; offset += chunkSize)
                        encoder.Write(data.AsSpan(offset, Math.Min(chunkSize, len - offset)));
                    encoder.Complete();
                }

                Assert.Equal(Encoding.UTF8.GetBytes(expected), output.ToArray());
            }
        }
    }

    [Fact]
    public void Write_WithArrayOverload_MatchesSpanOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xABC, 500);

        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output))
        {
            for (int i = 0; i < data.Length; i++)
                encoder.Write(data, i, 1);
            encoder.Complete();
        }

        Assert.Equal(Encoding.UTF8.GetBytes(Base122.Encode(data)), output.ToArray());
    }

    [Fact]
    public void Write_InterleavedWithFlush_ProducesContinuousStream()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x5150, 5000);

        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output))
        {
            for (int offset = 0; offset < data.Length; offset += 777)
            {
                encoder.Write(data.AsSpan(offset, Math.Min(777, data.Length - offset)));
                encoder.Flush();
                Assert.True(output.Position > 0, "Flush should push buffered output to the stream.");
            }
            encoder.Complete();
        }

        Assert.Equal(Encoding.UTF8.GetBytes(Base122.Encode(data)), output.ToArray());
    }

    [Fact]
    public void Dispose_ImplicitlyCompletes()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x99, 100);
        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output))
            encoder.Write(data);

        // Encoder disposed without Complete: output must still be a full valid encoding.
        string decoded = Encoding.UTF8.GetString(output.ToArray());
        Assert.Equal(data, Base122.Decode(decoded));
    }

    [Fact]
    public void Write_AfterComplete_Throws()
    {
        using var encoder = new Base122Encoder(new MemoryStream());
        encoder.Complete();
        Assert.Throws<InvalidOperationException>(() => encoder.Write([1, 2, 3]));
        Assert.Throws<InvalidOperationException>(() => encoder.Write([1, 2, 3], 0, 3));
    }

    [Fact]
    public void Complete_IsIdempotent()
    {
        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output);
        encoder.Write("data"u8);
        encoder.Complete();
        long lengthAfterFirstComplete = output.Length;
        encoder.Complete();
        Assert.Equal(lengthAfterFirstComplete, output.Length);
    }

    [Fact]
    public void LeaveOpen_KeepsOutputStreamUsable()
    {
        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: true))
            encoder.Write("test"u8);

        // The stream must still be readable after disposing the encoder.
        Assert.True(output.CanRead);
        output.Position = 0;
        using var decoder = new Base122Decoder(output, leaveOpen: true);
        var buffer = new byte[16];
        int read = decoder.Read(buffer);
        Assert.Equal("test"u8, buffer.AsSpan(0, read).ToArray());
    }

    [Fact]
    public void LeaveOpen_False_DisposesUnderlyingStream()
    {
        var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: false))
            encoder.Write("test"u8);

        Assert.False(output.CanRead);
    }

    [Fact]
    public void NullStream_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new Base122Encoder(null!));
    }

    [Fact]
    public void Write_InvalidArrayBounds_Throw()
    {
        using var encoder = new Base122Encoder(new MemoryStream());
        Assert.Throws<ArgumentNullException>(() => encoder.Write((byte[])null!, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Write(new byte[4], -1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Write(new byte[4], 0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Write(new byte[4], 3, 2));
    }

    [Fact]
    public void Write_AfterDispose_Throws()
    {
        var encoder = new Base122Encoder(new MemoryStream());
        encoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => encoder.Write([1]));
        Assert.Throws<ObjectDisposedException>(() => encoder.Flush());
        Assert.Throws<ObjectDisposedException>(() => encoder.Complete());
    }

    private static Stream GetStream(Base122Encoder _) => throw new InvalidOperationException();
}

/// <summary>Tests for the streaming <see cref="Base122Decoder"/>.</summary>
public class Base122DecoderStreamTests
{
    [Fact]
    public void Read_MatchesStaticDecoding_ForVariousBufferSizes()
    {
        foreach (int len in new[] { 1, 2, 3, 7, 8, 9, 14, 15, 27, 28, 29, 55, 56, 57, 100, 255, 256, 1000 })
        {
            byte[] data = TestHelpers.RandomBytesArray((ulong)len + 12345, len);
            byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data));

            foreach (int bufferSize in new[] { 1, 2, 3, 5, 7, 8, 13, 64, 4096 })
            {
                using var decoder = new Base122Decoder(new MemoryStream(encoded));
                var decoded = new byte[len];
                int filled = 0, read;
                while ((read = decoder.Read(decoded.AsSpan(filled, Math.Min(bufferSize, len - filled)))) > 0)
                    filled += read;

                Assert.Equal(len, filled);
                Assert.Equal(data, decoded);
            }
        }
    }

    [Fact]
    public void Read_WithTrickleInputStream_DecodesCorrectly()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xF00D, 2048);
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data));

        using var decoder = new Base122Decoder(new TrickleStream(new MemoryStream(encoded), 3));
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while (filled < data.Length &&
               (read = decoder.Read(decoded.AsSpan(filled, Math.Min(17, data.Length - filled)))) > 0)
            filled += read;

        Assert.Equal(data.Length, filled);
        Assert.Equal(data, decoded);
    }

    [Fact]
    public void Read_WithNonSeekableStream_DecodesCorrectly()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xBEEF, 1000);
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data));

        using var decoder = new Base122Decoder(new NonSeekableStream(new MemoryStream(encoded)));
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while (filled < data.Length &&
               (read = decoder.Read(decoded.AsSpan(filled, Math.Min(32, data.Length - filled)))) > 0)
            filled += read;

        Assert.Equal(data, decoded.AsSpan(0, filled).ToArray());
    }

    [Fact]
    public void Read_BufferSmallerThanAccumulator_DrainsBeforeReadingInput()
    {
        // 100 zero bytes encode to 57 escape characters plus a shortened
        // escape. Each escape pushes 14 bits, so with a one-byte read buffer
        // the accumulator repeatedly holds a complete byte between calls; the
        // decoder must emit it before consuming more input.
        byte[] data = new byte[100];
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data));

        using var decoder = new Base122Decoder(new MemoryStream(encoded));
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while (filled < data.Length &&
               (read = decoder.Read(decoded.AsSpan(filled, data.Length - filled))) > 0)
            filled += read;

        Assert.Equal(data.Length, filled);
        Assert.Equal(data, decoded);
    }

    [Fact]
    public void ReadByte_ReturnsBytesThenMinusOne()
    {
        byte[] data = [0x02];
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data)); // 01 de 80

        using var decoder = new Base122Decoder(new MemoryStream(encoded));
        Assert.Equal(0x02, decoder.ReadByte());
        Assert.Equal(-1, decoder.ReadByte());
    }

    [Fact]
    public void Read_AtEndOfStream_ReturnsZero()
    {
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode("hello world"u8));
        using var decoder = new Base122Decoder(new MemoryStream(encoded));

        var buffer = new byte[64];
        int total = 0, read;
        while ((read = decoder.Read(buffer)) > 0)
            total += read;

        Assert.Equal("hello world"u8.Length, total);
        Assert.Equal(0, decoder.Read(buffer));
        Assert.Equal(0, decoder.Read(buffer));
    }

    [Fact]
    public void Read_EmptyBuffer_ReturnsZero()
    {
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode("hello"u8));
        using var decoder = new Base122Decoder(new MemoryStream(encoded));
        Assert.Equal(0, decoder.Read(Array.Empty<byte>()));
        Assert.Equal(0, decoder.Read(new byte[10], 5, 0));
    }

    [Fact]
    public void Read_TruncatedEscape_Throws()
    {
        // The encoding of 100 zero bytes ends with a shortened escape whose
        // UTF-8 form is DE 80; removing the final byte truncates the escape.
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(new byte[100]));
        Assert.Equal(0x80, encoded[^1]);

        using var decoder = new Base122Decoder(new MemoryStream(encoded[..^1]));
        Assert.Throws<InvalidDataException>(() => decoder.Read(new byte[128]));
    }

    [Fact]
    public void Read_InvalidLeadByte_Throws()
    {
        // 0xC4 maps to no legal illegal-index, and neither do 0x80..0xC1 or 0xE0+.
        foreach (byte[] bad in new[] { new byte[] { 0xC4 }, new byte[] { 0x80 }, new byte[] { 0xE0, 0xA0, 0x80 } })
        {
            using var decoder = new Base122Decoder(new MemoryStream(bad));
            Assert.Throws<InvalidDataException>(() => decoder.Read(new byte[16]));
        }
    }

    [Fact]
    public void Read_InvalidContinuationByte_Throws()
    {
        // 0xC2 is a legal escape lead, but 'A' (0x41) is not a continuation byte.
        using var decoder = new Base122Decoder(new MemoryStream([0xC2, 0x41]));
        Assert.Throws<InvalidDataException>(() => decoder.Read(new byte[16]));
    }

    [Fact]
    public void Read_InvalidArrayBounds_Throw()
    {
        using var decoder = new Base122Decoder(new MemoryStream([0x41]));
        Assert.Throws<ArgumentNullException>(() => decoder.Read((byte[])null!, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Read(new byte[4], -1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Read(new byte[4], 0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Read(new byte[4], 3, 2));
    }

    [Fact]
    public void Read_AfterDispose_Throws()
    {
        var decoder = new Base122Decoder(new MemoryStream([0x41]));
        decoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => decoder.Read(new byte[4]));
        Assert.Throws<ObjectDisposedException>(() => decoder.ReadByte());
    }

    [Fact]
    public void NullStream_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new Base122Decoder(null!));
    }
}

/// <summary>End-to-end piping between the streaming encoder and decoder.</summary>
public class StreamPipelineTests
{
    [Fact]
    public void EncoderToDecoder_RoundTripsWithRandomChunking()
    {
        foreach (int len in new[] { 0, 1, 7, 8, 57, 100, 513, 2000 })
        {
            byte[] data = TestHelpers.RandomBytesArray((ulong)len + 77, len);

            using var pipe = new MemoryStream();
            using (var encoder = new Base122Encoder(pipe, leaveOpen: true))
            {
                int offset = 0;
                while (offset < len)
                {
                    int chunk = (int)(TestHelpers.NextU64(ref TestHelpers.Seed) % 37) + 1;
                    chunk = Math.Min(chunk, len - offset);
                    encoder.Write(data.AsSpan(offset, chunk));
                    offset += chunk;
                }
                encoder.Complete();
            }

            pipe.Position = 0;
            using var decoder = new Base122Decoder(pipe, leaveOpen: true);
            var decoded = new byte[len];
            int filled = 0, read;
            while ((read = decoder.Read(decoded.AsSpan(filled, len - filled))) > 0)
                filled += read;

            Assert.Equal(len, filled);
            Assert.Equal(data, decoded);
        }
    }

    [Fact]
    public void EncoderToDecoder_RoundTripsIllegalHeavyData()
    {
        byte[] data = new byte[1000];
        byte[] illegals = TestHelpers.IllegalBytes;
        for (int i = 0; i < data.Length; i++)
            data[i] = i % 2 == 0 ? illegals[i % illegals.Length] : (byte)(i & 0xFF);

        using var pipe = new MemoryStream();
        using (var encoder = new Base122Encoder(pipe, leaveOpen: true))
        {
            encoder.Write(data);
            encoder.Complete();
        }

        Assert.Equal(Encoding.UTF8.GetBytes(Base122.Encode(data)), pipe.ToArray());

        pipe.Position = 0;
        using var decoder = new Base122Decoder(pipe, leaveOpen: true);
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while ((read = decoder.Read(decoded.AsSpan(filled))) > 0)
            filled += read;

        Assert.Equal(data, decoded);
    }

    [Fact]
    public void Decoder_MatchesStaticVectors_AcrossTheWholeRustCorpus()
    {
        foreach (var (dataHex, encodedHex) in RustVectorData.Vectors.Take(2000))
        {
            byte[] expected = TestHelpers.FromHex(dataHex);
            byte[] encoded = TestHelpers.FromHex(encodedHex);

            using var decoder = new Base122Decoder(new MemoryStream(encoded));
            Span<byte> buffer = expected.Length == 0 ? Span<byte>.Empty : new byte[expected.Length];
            int filled = 0, read;
            while (!buffer.IsEmpty && (read = decoder.Read(buffer[filled..])) > 0)
                filled += read;

            Assert.Equal(expected.Length, filled);
            Assert.True(expected.AsSpan().SequenceEqual(buffer[..filled]));
        }
    }
}
