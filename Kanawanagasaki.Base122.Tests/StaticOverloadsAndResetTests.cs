using System.Text;

using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>Reusing one <see cref="Base122Encoder"/> instance for multiple encoded chunks via <see cref="Base122Encoder.Reset"/>.</summary>
public class EncoderResetTests
{
    [Fact]
    public void Reset_AfterComplete_AllowsWritingAnotherChunk()
    {
        byte[] chunk1 = TestHelpers.RandomBytesArray(0x1D1, 100);
        byte[] chunk2 = TestHelpers.RandomBytesArray(0x2D2, 130);

        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output, leaveOpen: true);
        encoder.Write(chunk1);
        encoder.Complete();

        encoder.Reset(); // revive the same instance for the next chunk

        encoder.Write(chunk2);
        encoder.Complete();

        byte[] encoded = output.ToArray();
        byte[] expected = [.. Base122Encoder.EncodeToUtf8(chunk1), .. Base122Encoder.EncodeToUtf8(chunk2)];
        Assert.Equal(expected, encoded);

        Assert.Equal(chunk1, Base122Decoder.DecodeFromUtf8(encoded[..Base122Encoder.EncodeToUtf8(chunk1).Length]));
    }

    [Fact]
    public void Reset_SupportsInterleavedPlainTextBorders()
    {
        // The scenario: text -> base122 -> border -> base122 -> text ...
        byte[] chunk1 = TestHelpers.RandomBytesArray(0xB0E1, 500);
        byte[] chunk2 = TestHelpers.RandomBytesArray(0xB0E2, 700);
        byte[] border = "\n---\n"u8.ToArray();

        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: true))
        {
            output.Write("header: "u8);
            encoder.Write(chunk1);
            encoder.Complete();
            encoder.Reset();

            output.Write(border);

            encoder.Write(chunk2);
            encoder.Complete();
            encoder.Reset();

            output.Write("footer"u8);
        }

        byte[] raw = output.ToArray();
        string text = Encoding.UTF8.GetString(raw); // whole message is valid UTF-8

        // The border can never occur inside encoded chunks, so splitting is
        // unambiguous: one border -> two segments.
        string[] parts = text.Split("\n---\n");
        Assert.Equal(2, parts.Length);
        Assert.StartsWith("header: ", parts[0]);
        Assert.EndsWith("footer", parts[1]);

        byte[] payload1 = Encoding.UTF8.GetBytes(parts[0]["header: ".Length..]);
        byte[] payload2 = Encoding.UTF8.GetBytes(parts[1][..^"footer".Length]);
        Assert.Equal(chunk1, Base122Decoder.DecodeFromUtf8(payload1));
        Assert.Equal(chunk2, Base122Decoder.DecodeFromUtf8(payload2));
    }

    [Fact]
    public void Reset_ManyChunksInARow_MatchesPerChunkEncodings()
    {
        byte[][] chunks = Enumerable.Range(0, 10)
            .Select(i => TestHelpers.RandomBytesArray(0x9E7 + (uint)i, 100 + i * 37))
            .ToArray();

        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: true))
        {
            foreach (byte[] chunk in chunks)
            {
                encoder.Write(chunk);
                encoder.Complete();
                encoder.Reset();
            }
        }

        byte[] expected = chunks.SelectMany(c => Base122Encoder.EncodeToUtf8(c)).ToArray();
        Assert.Equal(expected, output.ToArray());
    }

    [Fact]
    public void Reset_DiscardsUnfinalizedBitState_ButFlushesBufferedOutput()
    {
        byte[] chunk = TestHelpers.RandomBytesArray(0xD15C, 1000);

        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output, leaveOpen: true);
        encoder.Write(chunk);
        encoder.Reset(); // not completed: buffered bytes are flushed, trailing bits dropped

        // Whatever was flushed decodes cleanly on its own (padding bits dropped
        // by the decoder are exactly the unfinalized remainder).
        byte[] raw = output.ToArray();
        byte[] decoded = Base122Decoder.DecodeFromUtf8(raw);
        Assert.Equal(chunk[..decoded.Length], decoded);
    }

    [Fact]
    public void Reset_AfterDispose_Throws()
    {
        var encoder = new Base122Encoder(new MemoryStream());
        encoder.Dispose();
        Assert.Throws<ObjectDisposedException>(encoder.Reset);
    }

    [Fact]
    public void Reset_NeverClosesTheUnderlyingStream()
    {
        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output, leaveOpen: true);
        encoder.Write("one"u8);
        encoder.Complete();
        encoder.Reset();
        Assert.True(output.CanWrite);
        Assert.Equal(0, output.Position % 1); // stream untouched and usable
    }
}

/// <summary>The new static Stream-destination/source overloads.</summary>
public class StaticStreamOverloadTests
{
    [Fact]
    public void EncodeToUtf8_ToStream_MatchesArrayOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x57EA4, 50_000);

        using var output = new MemoryStream();
        int written = Base122Encoder.EncodeToUtf8(data, output);

        Assert.Equal(Base122Encoder.EncodeToUtf8(data), output.ToArray());
        Assert.Equal(written, output.Length);
    }

    [Fact]
    public void EncodeToUtf8_ToStream_EmptyData_WritesNothing()
    {
        using var output = new MemoryStream();
        int written = Base122Encoder.EncodeToUtf8(ReadOnlySpan<byte>.Empty, output);

        Assert.Equal(0, written);
        Assert.Equal(0, output.Length);
    }

    [Fact]
    public void EncodeToUtf8_ToNonWritableStream_Throws()
    {
        using var stream = new ReadOnlyStream();
        Assert.Throws<NotSupportedException>(() => Base122Encoder.EncodeToUtf8("data"u8, stream));
    }

    [Fact]
    public void DecodeFromUtf8_ToStream_MatchesArrayOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x0F1CE, 30_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);

        using var destination = new MemoryStream();
        int written = Base122Decoder.DecodeFromUtf8(encoded, destination);

        Assert.Equal(data, destination.ToArray());
        Assert.Equal(data.Length, written);
    }

    [Fact]
    public void DecodeFromUtf8_ToStream_EmptyInput_WritesNothing()
    {
        using var destination = new MemoryStream();
        int written = Base122Decoder.DecodeFromUtf8(ReadOnlySpan<byte>.Empty, destination);

        Assert.Equal(0, written);
        Assert.Equal(0, destination.Length);
    }

    [Fact]
    public void DecodeFromUtf8_ToNonWritableStream_Throws()
    {
        using var stream = new ReadOnlyStream();
        Assert.Throws<NotSupportedException>(() => Base122Decoder.DecodeFromUtf8("data"u8, stream));
    }

    [Fact]
    public void DecodeFromUtf8_ToStream_InvalidInput_ThrowsFormat()
    {
        using var destination = new MemoryStream();
        Assert.Throws<FormatException>(() => Base122Decoder.DecodeFromUtf8([0xC4, 0x80], destination));
    }

    private sealed class ReadOnlyStream : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => 0;
        public override long Position { get => 0; set { } }
        public override int Read(byte[] buffer, int offset, int count) => 0;
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class WriteOnlyStream : Stream
    {
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => 0;
        public override long Position { get => 0; set { } }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) { }
    }
}

/// <summary>The new static <see cref="IBufferWriter{Byte}"/> overloads.</summary>
public class StaticBufferWriterTests
{
    [Fact]
    public void EncodeToUtf8_ToBufferWriter_MatchesArrayOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xB1FF, 20_000);
        var writer = new TestBufferWriter();

        int written = Base122Encoder.EncodeToUtf8(data, writer);

        Assert.Equal(Base122Encoder.EncodeToUtf8(data), writer.ToArray());
        Assert.Equal(writer.Written, written);
    }

    [Fact]
    public void EncodeToUtf8_ToBufferWriter_EmptyData_WritesNothingAndRequestsNothing()
    {
        var writer = new TestBufferWriter();

        int written = Base122Encoder.EncodeToUtf8(ReadOnlySpan<byte>.Empty, writer);

        Assert.Equal(0, written);
        Assert.Equal(0, writer.GetSpanCalls);
        Assert.Equal(0, writer.Written);
    }

    [Fact]
    public void DecodeFromUtf8_ToBufferWriter_MatchesArrayOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xB1FF, 20_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);
        var writer = new TestBufferWriter();

        int written = Base122Decoder.DecodeFromUtf8(encoded, writer);

        Assert.Equal(data, writer.ToArray());
        Assert.Equal(data.Length, written);
        Assert.Equal(data.Length, writer.Written);
    }

    [Fact]
    public void DecodeFromUtf8_ToBufferWriter_InvalidInput_ThrowsFormat()
    {
        var writer = new TestBufferWriter();
        Assert.Throws<FormatException>(() => Base122Decoder.DecodeFromUtf8([0xC4, 0x80], writer));
    }

    [Fact]
    public void DecodeFromUtf8_ToBufferWriter_EmptyInput_WritesNothing()
    {
        var writer = new TestBufferWriter();
        int written = Base122Decoder.DecodeFromUtf8(ReadOnlySpan<byte>.Empty, writer);
        Assert.Equal(0, written);
        Assert.Equal(0, writer.Written);
    }
}

/// <summary>The new static async Stream overloads.</summary>
public class StaticAsyncOverloadTests
{
    [Fact]
    public async Task EncodeToUtf8Async_ToStream_MatchesSyncOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xA5A5C, 100_000);

        using var output = new MemoryStream();
        await Base122Encoder.EncodeToUtf8Async(data, output);

        Assert.Equal(Base122Encoder.EncodeToUtf8(data), output.ToArray());
    }

    [Fact]
    public async Task AsyncOverloads_OverYieldingStreams_ProduceValidResults()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xF0F0, 20_000);

        using var encodedStream = new MemoryStream();
        await Base122Encoder.EncodeToUtf8Async(data, new YieldAsyncStream(encodedStream));
        byte[] encoded = encodedStream.ToArray();
        Assert.Equal(Base122Encoder.EncodeToUtf8(data), encoded);

        using var decodedStream = new MemoryStream();
        await Base122Decoder.DecodeFromUtf8Async(encoded, new YieldAsyncStream(decodedStream));
        Assert.Equal(data, decodedStream.ToArray());
    }

    [Fact]
    public async Task CanceledToken_ThrowsOperationCanceled()
    {
        using var canceled = new CancellationTokenSource();
        await canceled.CancelAsync();

        using var output = new MemoryStream();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => Base122Encoder.EncodeToUtf8Async(new byte[100], output, canceled.Token));
    }

    [Fact]
    public async Task DecodeFromUtf8Async_ToStream_MatchesSyncOverload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xDE57, 80_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);

        using var destination = new MemoryStream();
        int written = await Base122Decoder.DecodeFromUtf8Async(encoded, destination);

        Assert.Equal(data, destination.ToArray());
        Assert.Equal(data.Length, written);
    }

    [Fact]
    public async Task DecodeFromUtf8Async_ToYieldingStream_UsesRealAsyncWrites()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xDE57, 20_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);

        using var backing = new MemoryStream();
        int written = await Base122Decoder.DecodeFromUtf8Async(encoded, new YieldAsyncStream(backing));

        Assert.Equal(data, backing.ToArray());
        Assert.Equal(data.Length, written);
    }

    [Fact]
    public async Task DecodeFromUtf8Async_ToStream_EmptyInput_WritesNothing()
    {
        using var destination = new MemoryStream();
        int written = await Base122Decoder.DecodeFromUtf8Async(ReadOnlyMemory<byte>.Empty, destination);

        Assert.Equal(0, written);
        Assert.Equal(0, destination.Length);
    }

    [Fact]
    public async Task DecodeFromUtf8Async_CanceledToken_ThrowsOperationCanceled()
    {
        using var canceled = new CancellationTokenSource();
        await canceled.CancelAsync();
        using var destination = new MemoryStream();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => Base122Decoder.DecodeFromUtf8Async(new byte[100], destination, canceled.Token));
    }

    [Fact]
    public async Task EncodeToUtf8Async_EmptyData_WritesNothing()
    {
        using var output = new MemoryStream();
        await Base122Encoder.EncodeToUtf8Async(ReadOnlyMemory<byte>.Empty, output);
        Assert.Equal(0, output.Length);
    }
}
