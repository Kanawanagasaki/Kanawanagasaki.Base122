using System.Text;

using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>
/// A stream that forces a real asynchronous continuation on every
/// <see cref="ReadAsync"/> (via <see cref="Task.Yield"/>), so async decode
/// paths are exercised against genuinely incomplete tasks.
/// </summary>
public sealed class YieldAsyncStream(Stream inner) : Stream
{
    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => inner.Length;
    public override long Position { get => inner.Position; set => inner.Position = value; }

    public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        return await inner.ReadAsync(buffer, cancellationToken);
    }

    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

/// <summary>The <see cref="Stream"/> contract of the encoder and decoder.</summary>
public class StreamContractTests
{
    [Fact]
    public void Encoder_IsWriteOnlyNonSeekableStream()
    {
        using var encoder = new Base122Encoder(new MemoryStream());

        Assert.True(encoder.CanWrite);
        Assert.False(encoder.CanRead);
        Assert.False(encoder.CanSeek);

        Assert.Throws<NotSupportedException>(() => encoder.Length);
        Assert.Throws<NotSupportedException>(() => encoder.Position);
        Assert.Throws<NotSupportedException>(() => encoder.Position = 0);
        Assert.Throws<NotSupportedException>(() => encoder.Seek(0, SeekOrigin.Begin));
        Assert.Throws<NotSupportedException>(() => encoder.SetLength(0));
        Assert.Throws<NotSupportedException>(() => encoder.Read(new byte[8], 0, 8));
        Assert.Throws<NotSupportedException>(() => encoder.Read(new byte[8]));
    }

    [Fact]
    public void Decoder_IsReadOnlyNonSeekableStream()
    {
        using var decoder = new Base122Decoder(new MemoryStream([0x41]));

        Assert.True(decoder.CanRead);
        Assert.False(decoder.CanWrite);
        Assert.False(decoder.CanSeek);

        Assert.Throws<NotSupportedException>(() => decoder.Length);
        Assert.Throws<NotSupportedException>(() => decoder.Position);
        Assert.Throws<NotSupportedException>(() => decoder.Position = 0);
        Assert.Throws<NotSupportedException>(() => decoder.Seek(0, SeekOrigin.Begin));
        Assert.Throws<NotSupportedException>(() => decoder.SetLength(0));
        Assert.Throws<NotSupportedException>(() => decoder.Write([1, 2, 3], 0, 3));
        Assert.Throws<NotSupportedException>(() => decoder.Write([1, 2, 3]));
        Assert.Throws<NotSupportedException>(() => decoder.WriteByte(1));
    }

    [Fact]
    public void Encoder_WorksWithStreamApiShapes()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xC072AC7, 300);
        using var output = new MemoryStream();

        Stream encoder = new Base122Encoder(output, leaveOpen: true);
        encoder.Write(data, 0, data.Length);          // Stream abstract shape
        encoder.WriteByte(0xAB);                      // WriteByte
        encoder.Flush();
        ((Base122Encoder)encoder).Complete();
        encoder.Dispose();

        byte[] expected = Base122.EncodeToUtf8(data.Append((byte)0xAB).ToArray());
        Assert.Equal(expected, output.ToArray());
    }

    [Fact]
    public void Decoder_ReadsThroughStreamReader()
    {
        byte[] text = "The quick brown fox jumps over the lazy dog."u8.ToArray();
        byte[] encoded = Base122.EncodeToUtf8(text);

        using var decoder = new Base122Decoder(new MemoryStream(encoded));
        using var reader = new StreamReader(decoder);

        Assert.Equal(Encoding.UTF8.GetString(text), reader.ReadToEnd());
    }
}

/// <summary>Synchronous piping: <see cref="Stream.CopyTo(Stream)"/> in both directions.</summary>
public class SyncPipeTests
{
    [Fact]
    public void RawStream_CopyTo_Encoder_WritesValidEncoding()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x91DE, 50_000);

        using var raw = new MemoryStream(data);
        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: true))
        {
            raw.CopyTo(encoder);   // pipe: raw -> encoder
            encoder.Complete();
        }

        Assert.Equal(Base122.EncodeToUtf8(data), output.ToArray());
    }

    [Fact]
    public void Decoder_CopyTo_Destination_ProducesDecodedData()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x91DE, 50_000);
        byte[] encoded = Base122.EncodeToUtf8(data);

        using var input = new MemoryStream(encoded);
        using var decoder = new Base122Decoder(input, leaveOpen: true);
        using var destination = new MemoryStream();
        decoder.CopyTo(destination);   // pipe: decoder -> destination

        Assert.Equal(data, destination.ToArray());
    }

    [Fact]
    public void FullPipeline_RawToEncodedToDecoded_ThroughCopyTo()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x9191, 100_000);

        using var raw = new MemoryStream(data);
        using var encodedStream = new MemoryStream();
        using (var encoder = new Base122Encoder(encodedStream, leaveOpen: true))
        {
            raw.CopyTo(encoder);
            encoder.Complete();
        }

        Assert.Equal(Base122.EncodeToUtf8(data), encodedStream.ToArray());

        encodedStream.Position = 0;
        using var decodedStream = new MemoryStream();
        using (var decoder = new Base122Decoder(encodedStream, leaveOpen: true))
            decoder.CopyTo(decodedStream);

        Assert.Equal(data, decodedStream.ToArray());
    }
}

/// <summary>Asynchronous API and piping: WriteAsync/ReadAsync, CopyToAsync, DisposeAsync, cancellation.</summary>
public class AsyncStreamTests
{
    public static IEnumerable<object[]> ChunkSizes() =>
        new[] { 1, 3, 7, 8, 13, 100, 1000, 8192 }.Select(c => new object[] { c });

    [Theory]
    [MemberData(nameof(ChunkSizes))]
    public async Task WriteAsync_Chunked_MatchesStaticEncoding(int chunkSize)
    {
        byte[] data = TestHelpers.RandomBytesArray(0xA5A5, 3000);

        using var output = new MemoryStream();
        await using (var encoder = new Base122Encoder(output, leaveOpen: true))
        {
            for (int offset = 0; offset < data.Length; offset += chunkSize)
                await encoder.WriteAsync(data.AsMemory(offset, Math.Min(chunkSize, data.Length - offset)));
            await encoder.CompleteAsync();
        }

        Assert.Equal(Base122.EncodeToUtf8(data), output.ToArray());
    }

    [Theory]
    [MemberData(nameof(ChunkSizes))]
    public async Task ReadAsync_Chunked_MatchesData(int chunkSize)
    {
        byte[] data = TestHelpers.RandomBytesArray(0x5A5A, 3000);
        byte[] encoded = Base122.EncodeToUtf8(data);

        await using var decoder = new Base122Decoder(new MemoryStream(encoded), leaveOpen: true);
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while (filled < decoded.Length &&
               (read = await decoder.ReadAsync(decoded.AsMemory(filled, Math.Min(chunkSize, decoded.Length - filled)))) > 0)
            filled += read;

        Assert.Equal(data.Length, filled);
        Assert.Equal(data, decoded);
        Assert.Equal(0, await decoder.ReadAsync(decoded.AsMemory(0)));
    }

    [Fact]
    public async Task ReadAsync_OverYieldingStream_ForcesRealAwaits()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xE11A, 20_000);
        byte[] encoded = Base122.EncodeToUtf8(data);

        await using var decoder = new Base122Decoder(new YieldAsyncStream(new MemoryStream(encoded)));
        var decoded = new byte[data.Length];
        int filled = 0, read;
        while (filled < data.Length &&
               (read = await decoder.ReadAsync(decoded.AsMemory(filled, Math.Min(997, data.Length - filled)))) > 0)
            filled += read;

        Assert.Equal(data, decoded);
    }

    [Fact]
    public async Task CopyToAsync_PipesBothDirections()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xD1D1, 100_000);

        // Encode: raw -> encoder.
        using var raw = new MemoryStream(data);
        using var encodedStream = new MemoryStream();
        await using (var encoder = new Base122Encoder(encodedStream, leaveOpen: true))
        {
            await raw.CopyToAsync(encoder);
            await encoder.CompleteAsync();
        }
        Assert.Equal(Base122.EncodeToUtf8(data), encodedStream.ToArray());

        // Decode: decoder -> destination.
        encodedStream.Position = 0;
        using var destination = new MemoryStream();
        await using (var decoder = new Base122Decoder(encodedStream, leaveOpen: true))
            await decoder.CopyToAsync(destination);

        Assert.Equal(data, destination.ToArray());
    }

    [Fact]
    public async Task MixedSyncAndAsync_Writes_ProduceValidEncoding()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xE1E1, 5000);

        using var output = new MemoryStream();
        await using (var encoder = new Base122Encoder(output, leaveOpen: true))
        {
            encoder.Write(data.AsSpan(0, 1000));                 // sync
            await encoder.WriteAsync(data.AsMemory(1000, 2000)); // async
            for (int offset = 3000; offset < data.Length; offset += 555)
                await encoder.WriteAsync(data.AsMemory(offset, Math.Min(555, data.Length - offset)));
            await encoder.CompleteAsync();
        }

        Assert.Equal(Base122.EncodeToUtf8(data), output.ToArray());
    }

    [Fact]
    public async Task CanceledToken_ThrowsOperationCanceled()
    {
        using var canceled = new CancellationTokenSource();
        await canceled.CancelAsync();

        await using var encoder = new Base122Encoder(new MemoryStream(), leaveOpen: true);
        await using var decoder = new Base122Decoder(new MemoryStream([0x41]), leaveOpen: true);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => encoder.WriteAsync("data"u8.ToArray().AsMemory(), canceled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => encoder.WriteAsync(new byte[10], 0, 10, canceled.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => decoder.ReadAsync(new byte[10], canceled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => decoder.ReadAsync(new byte[10].AsMemory(), canceled.Token).AsTask());

        // And after cancellation, a clean token still works.
        var buffer = new byte[16];
        Assert.True(await decoder.ReadAsync(buffer) >= 0);
        await encoder.WriteAsync("data"u8.ToArray().AsMemory());
        await encoder.CompleteAsync();
    }

    [Fact]
    public async Task DisposeAsync_FinalizesAndDisposesInner()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xE1E2, 1000);

        var output = new MemoryStream();
        await using (var encoder = new Base122Encoder(output))
            await encoder.WriteAsync(data);

        // Encoder disposed without CompleteAsync: output must still be a full
        // valid encoding, and the inner stream must be disposed.
        Assert.Equal(Base122.EncodeToUtf8(data), output.ToArray());
        Assert.False(output.CanRead);
    }

    [Fact]
    public async Task DisposeAsync_LeaveOpen_KeepsInnerUsable()
    {
        var output = new MemoryStream();
        await using (var encoder = new Base122Encoder(output, leaveOpen: true))
            await encoder.WriteAsync("test"u8.ToArray());

        Assert.True(output.CanRead);
        output.Position = 0;
        await using var decoder = new Base122Decoder(output, leaveOpen: true);
        var buffer = new byte[16];
        int read = await decoder.ReadAsync(buffer);
        Assert.Equal("test"u8.ToArray(), buffer.AsSpan(0, read).ToArray());
    }

    [Fact]
    public async Task CompleteAsync_IsIdempotent()
    {
        await using var output = new MemoryStream();
        await using var encoder = new Base122Encoder(output, leaveOpen: true);
        await encoder.WriteAsync("data"u8.ToArray());
        await encoder.CompleteAsync();
        long lengthAfterFirstComplete = output.Length;
        await encoder.CompleteAsync();
        Assert.Equal(lengthAfterFirstComplete, output.Length);
    }

    [Fact]
    public async Task WriteAsync_AfterComplete_Throws()
    {
        await using var encoder = new Base122Encoder(new MemoryStream(), leaveOpen: true);
        await encoder.CompleteAsync();

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => encoder.WriteAsync("x"u8.ToArray().AsMemory()).AsTask());
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => encoder.WriteAsync(new byte[3], 0, 3, CancellationToken.None));
    }

    [Fact]
    public async Task ReadAsync_TruncatedEscape_ThrowsInvalidData()
    {
        byte[] encoded = Base122.EncodeToUtf8(new byte[100]);
        await using var decoder = new Base122Decoder(new MemoryStream(encoded[..^1]));
        await Assert.ThrowsAsync<InvalidDataException>(() => decoder.ReadAsync(new byte[128]).AsTask());
    }

    [Fact]
    public async Task AsyncPipeline_ThroughYieldingStreams_RoundTrips()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xE1E3, 50_000);

        using var raw = new MemoryStream(data);
        using var encodedStream = new MemoryStream();
        await using (var encoder = new Base122Encoder(encodedStream, leaveOpen: true))
        {
            await raw.CopyToAsync(encoder, 4096);
            await encoder.CompleteAsync();
        }

        Assert.Equal(Base122.EncodeToUtf8(data), encodedStream.ToArray());

        encodedStream.Position = 0;
        using var destination = new MemoryStream();
        await using (var decoder = new Base122Decoder(new YieldAsyncStream(encodedStream), leaveOpen: true))
            await decoder.CopyToAsync(destination, 4096);

        Assert.Equal(data, destination.ToArray());
    }
}
