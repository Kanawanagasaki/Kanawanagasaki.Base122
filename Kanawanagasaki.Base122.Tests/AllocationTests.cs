using System.Text;

using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>
/// Verifies the zero-allocation promises of the library: the array encoder
/// allocates only the resulting string, span APIs allocate nothing, and the
/// streaming classes do not allocate per write/read.
/// </summary>
public class AllocationTests
{
    private static long AllocatedBytes(Action action)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Warm up any lazy initialization.
        action();

        long before = GC.GetAllocatedBytesForCurrentThread();
        action();
        long after = GC.GetAllocatedBytesForCurrentThread();
        return after - before;
    }

    [Fact]
    public void Encode_ByteArray_AllocatesOnlyTheString()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xA11CE, 10_000);
        string expected = Base122.Encode(data);
        Assert.Equal(Base122.GetEncodedLength(data), expected.Length);

        long allocated = AllocatedBytes(() => Base122.Encode(data));
        // Exactly the string (~2 bytes per char) plus a small JIT/warmup margin.
        Assert.InRange(allocated, expected.Length * 2, expected.Length * 2 + 2048);
    }

    [Fact]
    public void Encode_LargeSpan_UsesPooledBufferOnly()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xB0B, 100_000);

        long allocated = AllocatedBytes(() => Base122.Encode(data.AsSpan()));
        // Only the result string (~228 KB) plus slack; no intermediate copies.
        Assert.InRange(allocated, 0, 300_000);
    }

    [Fact]
    public void Encode_SmallSpan_UsesStackalloc()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xC0FFEE, 200);

        long allocated = AllocatedBytes(() => Base122.Encode(data.AsSpan()));
        // Only the result string (~456 chars = 912 bytes) plus slack.
        Assert.InRange(allocated, 0, 2_048);
    }

    [Fact]
    public void TryEncodeAndTryDecode_AllocateNothing()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xD1CE, 10_000);
        char[] encodedBuffer = new char[Base122.GetMaxEncodedLength(data.Length)];
        byte[] decodedBuffer = new byte[Base122.GetMaxDecodedLength(encodedBuffer.Length)];

        long allocated = AllocatedBytes(() =>
        {
            Base122.TryEncode(data, encodedBuffer, out int charsWritten);
            Base122.TryDecode(encodedBuffer.AsSpan(0, charsWritten), decodedBuffer, out _);
        });

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void StreamEncoder_DoesNotAllocateAfterWarmup()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xE11E, 64 * 1024);

        using var output = new MemoryStream();
        using var encoder = new Base122Encoder(output, leaveOpen: true);

        long allocated = AllocatedBytes(() =>
        {
            output.Position = 0;
            output.SetLength(0);
            for (int offset = 0; offset < data.Length; offset += 1000)
                encoder.Write(data.AsSpan(offset, Math.Min(1000, data.Length - offset)));
            encoder.Flush();
        });

        // No per-write allocations beyond what the MemoryStream itself may buffer.
        Assert.InRange(allocated, 0, 1024);
    }

    [Fact]
    public void StreamDecoder_DoesNotAllocateAfterWarmup()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xFACE, 64 * 1024);
        byte[] encoded = Encoding.UTF8.GetBytes(Base122.Encode(data));

        using var input = new MemoryStream(encoded);
        using var decoder = new Base122Decoder(input, leaveOpen: true);
        byte[] buffer = new byte[8192];

        long allocated = AllocatedBytes(() =>
        {
            input.Position = 0;
            int read;
            while ((read = decoder.Read(buffer)) > 0) { }
        });

        Assert.Equal(0, allocated);
    }
}
