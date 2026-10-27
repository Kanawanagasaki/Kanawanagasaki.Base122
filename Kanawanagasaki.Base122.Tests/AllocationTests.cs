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
        string expected = Base122Encoder.Encode(data);
        Assert.Equal(Base122Encoder.GetEncodedLength(data), expected.Length);

        long allocated = AllocatedBytes(() => Base122Encoder.Encode(data));
        // Exactly the string (~2 bytes per char) plus a small JIT/warmup margin.
        Assert.InRange(allocated, expected.Length * 2, expected.Length * 2 + 2048);
    }

    [Fact]
    public void Encode_LargeSpan_AllocatesOnlyTheString()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xB0B, 100_000);
        string expected = Base122Encoder.Encode(data.AsSpan());

        long allocated = AllocatedBytes(() => Base122Encoder.Encode(data.AsSpan()));
        // Exact-length string.Create: only the result string (~228 KB).
        Assert.InRange(allocated, expected.Length * 2, expected.Length * 2 + 2048);
    }

    [Fact]
    public void Encode_SmallSpan_UsesStackalloc()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xC0FFEE, 200);

        long allocated = AllocatedBytes(() => Base122Encoder.Encode(data.AsSpan()));
        // Only the result string (~456 chars = 912 bytes) plus slack.
        Assert.InRange(allocated, 0, 2_048);
    }

    [Fact]
    public void TryEncodeAndTryDecode_AllocateNothing()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xD1CE, 10_000);
        char[] encodedBuffer = new char[Base122Encoder.GetMaxEncodedLength(data.Length)];
        byte[] decodedBuffer = new byte[Base122Decoder.GetMaxDecodedLength(encodedBuffer.Length)];

        long allocated = AllocatedBytes(() =>
        {
            Base122Encoder.TryEncode(data, encodedBuffer, out int charsWritten);
            Base122Decoder.TryDecode(encodedBuffer.AsSpan(0, charsWritten), decodedBuffer, out _);
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
        byte[] encoded = Encoding.UTF8.GetBytes(Base122Encoder.Encode(data));

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

    [Fact]
    public void StreamConstruction_AllocatesNothing()
    {
        var output = new MemoryStream();
        var input = new MemoryStream([0x41]);

        long allocated = AllocatedBytes(() =>
        {
            using var encoder = new Base122Encoder(output, leaveOpen: true);
            using var decoder = new Base122Decoder(input, leaveOpen: true);
        });

        // Only the two instance objects themselves (~64 + ~72 bytes) are
        // inherent to any class construction; the 8 KB staging buffers must
        // not exist until first use.
        Assert.True(allocated < 8192, $"Construction allocated {allocated} bytes.");
    }

    [Fact]
    public void StreamEncoder_BufferIsAllocatedOnce_OnFirstWrite()
    {
        byte[] data = new byte[64];
        var output = new MemoryStream(8192);

        using var encoder = new Base122Encoder(output, leaveOpen: true);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetAllocatedBytesForCurrentThread();
        encoder.Write(data);
        long firstWrite = GC.GetAllocatedBytesForCurrentThread() - before;

        long after = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++)
            encoder.Write(data);
        long subsequentWrites = GC.GetAllocatedBytesForCurrentThread() - after;

        // Exactly the 8 KB staging buffer (plus object header) on first use,
        // then nothing — the buffer is reused for the encoder's lifetime.
        Assert.InRange(firstWrite, 8192, 8192 + 64);
        Assert.Equal(0, subsequentWrites);
        encoder.Complete();
    }

    [Fact]
    public void StreamEncoder_WithCallerBuffer_AllocatesOnlyTheInstance()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xCA11E5, 10_000);
        byte[] expected = Base122Encoder.EncodeToUtf8(data);
        byte[] staging = new byte[8192];

        using var output = new MemoryStream(expected.Length + 64); // pre-sized
        long allocated = AllocatedBytes(() =>
        {
            // AllocatedBytes runs the action twice (warmup + measure); each
            // run must start from an empty stream.
            output.Position = 0;
            output.SetLength(0);
            using var encoder = new Base122Encoder(output, staging, leaveOpen: true);
            encoder.Write(data);
            encoder.Complete();
        });

        Assert.Equal(expected, output.ToArray());
        // Only the encoder instance itself (~64 bytes) is inherent; no staging
        // buffer and no intermediate copies are allocated.
        Assert.InRange(allocated, 0, 256);
        output.SetLength(0);
    }

    [Fact]
    public void StreamDecoder_WithCallerBuffer_AllocatesNothingEver()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xCA11E5, 10_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);
        byte[] staging = new byte[8192];

        var input = new MemoryStream(encoded);
        var decoded = new byte[data.Length];

        using var decoder = new Base122Decoder(input, staging, leaveOpen: true);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetAllocatedBytesForCurrentThread();
        int filled = 0, read;
        while ((read = decoder.Read(decoded.AsSpan(filled))) > 0)
            filled += read;
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(data, decoded);
        Assert.Equal(0, allocated);
    }

    /// <summary>
    /// Runs an async destination overload and measures its allocations,
    /// provided it completes synchronously (MemoryStream destinations do —
    /// every await inside completes synchronously). Returns the allocated
    /// bytes and the method result.
    /// </summary>
    private static (long Allocated, int Result) MeasureAsyncAllocations(ReadOnlyMemory<byte> payload, Func<ReadOnlyMemory<byte>, Stream, CancellationToken, Task<int>> call)
    {
        const int warmups = 3;
        for (int i = 0; i < warmups; i++)
        {
            using var sink = new MemoryStream(payload.Length * 2);
            Task<int> warm = call(payload, sink, CancellationToken.None);
            Assert.True(warm.IsCompleted, "Warm-up call did not complete synchronously.");
        }

        using var destination = new MemoryStream(payload.Length * 2);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetAllocatedBytesForCurrentThread();
        Task<int> task = call(payload, destination, CancellationToken.None);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(task.IsCompleted, "Measured call did not complete synchronously; thread-hopped allocations are not measured.");
        return (allocated, task.Result);
    }

    [Fact]
    public void EncodeToUtf8Async_DestinationStream_Small_DoesNotAllocate()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xA551AC, 8_000);

        (long allocated, int written) = MeasureAsyncAllocations(
            data, (p, s, ct) => Base122Encoder.EncodeToUtf8Async(p, s, ct));

        Assert.Equal(Base122Encoder.EncodeToUtf8(data).Length, written);
        // Only the completed Task<int> and state-machine remainders; the work
        // buffer is a pooled rent (pre-warmed by the calls above).
        Assert.InRange(allocated, 0, AsyncAllocBudget(512));
    }

#if DEBUG
    [Fact(Skip = "Strict async allocation budgets are Release-verified. Unoptimized Debug JIT boxes the per-await state machines of the streaming pump (~80 B per decoded byte), which is a Debug artifact, not library behavior.")]
#else
    [Fact]
#endif
    public void EncodeToUtf8Async_DestinationStream_Large_BoundedWorkingSet()
    {
        const int size = 10 * 1024 * 1024;
        byte[] data = TestHelpers.RandomBytesArray(0xB16B1E, size);

        (long allocated, int written) = MeasureAsyncAllocations(
            data, (p, s, ct) => Base122Encoder.EncodeToUtf8Async(p, s, ct));

        // Above the pooled-rent threshold the async encoder streams through a
        // pooled 64 KB staging buffer — nothing proportional to the input.
        Assert.Equal(Base122Encoder.EncodeToUtf8(data).Length, written);
        Assert.InRange(allocated, 0, AsyncAllocBudget(64 * 1024));
    }

    [Fact]
    public void DecodeFromUtf8Async_DestinationStream_Small_DoesNotAllocate()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xDEA5C, 8_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);

        (long allocated, int written) = MeasureAsyncAllocations(
            encoded, (p, s, ct) => Base122Decoder.DecodeFromUtf8Async(p, s, ct));

        Assert.Equal(data.Length, written);
        Assert.InRange(allocated, 0, AsyncAllocBudget(512));
    }

#if DEBUG
    [Fact(Skip = "Strict async allocation budgets are Release-verified. Unoptimized Debug JIT boxes the per-await state machines of the streaming pump (~80 B per decoded byte), which is a Debug artifact, not library behavior.")]
#else
    [Fact]
#endif
    public void DecodeFromUtf8Async_DestinationStream_Large_BoundedWorkingSet()
    {
        const int size = 10 * 1024 * 1024;
        byte[] encoded = Base122Encoder.EncodeToUtf8(TestHelpers.RandomBytesArray(0xB16B1D, size));

        (long allocated, int written) = MeasureAsyncAllocations(
            encoded, (p, s, ct) => Base122Decoder.DecodeFromUtf8Async(p, s, ct));

        // Above the pooled-rent threshold the async decoder streams through
        // pooled staging buffers — nothing proportional to the input.
        Assert.Equal(size, written);
        Assert.InRange(allocated, 0, AsyncAllocBudget(64 * 1024));
    }

    /// <summary>
    /// Allocation budgets for async APIs. The two large-input tests are
    /// skipped in Debug (see their Skip reason); these small-input budgets
    /// hold in both configurations because their await count is bounded.
    /// </summary>
    private static long AsyncAllocBudget(long releaseBudget) =>
#if DEBUG
        releaseBudget * 4;
#else
        releaseBudget;
#endif
}
