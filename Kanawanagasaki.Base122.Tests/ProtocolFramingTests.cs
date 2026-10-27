using System.Buffers;
using System.IO.Pipelines;
using System.Text;

using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>
/// Protocol scenario: plain-text header lines, two newlines, Base122 data,
/// two newlines, next frame — over a network-like pipe.
///
/// Writer: one long-lived <see cref="Base122Encoder"/> with a pooled staging
/// buffer, <see cref="Base122Encoder.Reset"/> between frames.
/// Reader: a long-lived read buffer, scanning for the "\n\n" terminator (LF
/// can never occur inside valid Base122 output) and spooling each data
/// segment into pooled memory, decoded with the static API.
/// </summary>
public class ProtocolFramingTests
{
    private static readonly byte[] TwoNewlines = "\n\n"u8.ToArray();

    // ---------------------------- writer side ----------------------------

    private static async Task ProduceAsync(Stream network, byte[][] headers, byte[][] chunks, CancellationToken ct)
    {
        byte[] staging = ArrayPool<byte>.Shared.Rent(8192);
        try
        {
            // One long-lived encoder for the whole session, Reset per frame.
            await using var encoder = new Base122Encoder(network, staging, leaveOpen: true);

            for (int i = 0; i < chunks.Length; i++)
            {
                await network.WriteAsync(headers[i], ct);   // header line(s), each \n-terminated
                await network.WriteAsync(TwoNewlines, ct);  // blank line ends the header block
                await encoder.WriteAsync(chunks[i], ct);
                await encoder.CompleteAsync(ct);
                await network.WriteAsync(TwoNewlines, ct);  // blank line ends the data block
                encoder.Reset();
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(staging);
        }
    }

    // ---------------------------- reader side ----------------------------

    /// <summary>
    /// Frame reader over a long-lived network stream: one long-lived read
    /// buffer, pooled per-segment spool, vectorized scan for the terminator.
    /// </summary>
    private sealed class FrameReader(Stream network)
    {
        private readonly byte[] _readBuffer = new byte[16 * 1024];
        private int _pos;
        private int _len;
        private bool _eof;

        /// <summary>
        /// Reads bytes until "\n\n" and returns the spooled segment (pooled —
        /// the caller must return it). Returns (null, 0) on clean end of
        /// stream between frames.
        /// </summary>
        private async Task<(byte[]? Buffer, int Length)> ReadSegmentAsync(CancellationToken ct)
        {
            byte[] spool = ArrayPool<byte>.Shared.Rent(16 * 1024);
            try
            {
                int len = 0;
                while (true)
                {
                    if (_pos >= _len)
                    {
                        if (_eof)
                        {
                            if (len == 0)
                                return (null, 0); // clean end between frames
                            throw new InvalidDataException("Protocol stream ended in the middle of a frame.");
                        }
                        _pos = 0;
                        _len = 0;
                        int read = await network.ReadAsync(_readBuffer.AsMemory(), ct);
                        if (read == 0)
                        {
                            _eof = true;
                            continue;
                        }
                        _len = read;
                    }

                    int available = _len - _pos;
                    if (len + available > spool.Length)
                    {
                        byte[] bigger = ArrayPool<byte>.Shared.Rent(Math.Max(spool.Length * 2, len + available));
                        Array.Copy(spool, bigger, len);
                        ArrayPool<byte>.Shared.Return(spool);
                        spool = bigger;
                    }

                    int appendedFrom = len;
                    Array.Copy(_readBuffer, _pos, spool, len, available);
                    len += available;
                    _pos = _len;

                    // Re-scan from one byte before the newly appended region so
                    // a terminator straddling two reads cannot be missed.
                    int searchFrom = Math.Max(0, appendedFrom - 1);
                    int idx = spool.AsSpan(searchFrom, len - searchFrom).IndexOf(TwoNewlines);
                    if (idx >= 0)
                    {
                        int terminatorAt = searchFrom + idx;

                        // Push back anything past the terminator — it belongs
                        // to the next frame's headers.
                        int leftover = len - (terminatorAt + TwoNewlines.Length);
                        if (leftover > 0)
                        {
                            Array.Copy(spool, terminatorAt + TwoNewlines.Length, _readBuffer, 0, leftover);
                            _pos = 0;
                            _len = leftover;
                        }

                        return (spool, terminatorAt);
                    }
                }
            }
            catch
            {
                ArrayPool<byte>.Shared.Return(spool);
                throw;
            }
        }

        /// <summary>Reads one frame. Returns null at clean end of stream.</summary>
        public async Task<(byte[] Headers, byte[] Data)?> ReadFrameAsync(CancellationToken ct)
        {
            (byte[]? Buffer, int Length) header = await ReadSegmentAsync(ct);
            if (header.Buffer is null)
                return null;
            byte[] headers = header.Buffer[..header.Length].ToArray();
            ArrayPool<byte>.Shared.Return(header.Buffer);

            (byte[]? Buffer, int Length) data = await ReadSegmentAsync(ct);
            if (data.Buffer is null)
                throw new InvalidDataException("Protocol stream ended between header and data blocks.");
            byte[] decoded;
            try
            {
                decoded = Base122Decoder.DecodeFromUtf8(data.Buffer.AsSpan(0, data.Length));
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(data.Buffer);
            }

            return (headers, decoded);
        }
    }

    // ---------------------------- the scenario ----------------------------

    private static byte[][] CreateHeaders(int count) =>
        Enumerable.Range(0, count).Select(i => Encoding.UTF8.GetBytes(i % 2 == 0
            ? $"X-Chunk: {i}\nContent-Encoding: base122"
            : $"X-Chunk: {i}")).ToArray();

    private static byte[][] CreateChunks(int count) => Enumerable.Range(0, count).Select(i =>
    {
        int len = new[] { 1, 7, 8, 100, 1000, 8192, 16384, 70_000, 150_000, 1000, 33, 5 }[i % 12];
        switch (i % 3)
        {
            case 0: return TestHelpers.RandomBytesArray(0xFEED00UL + (ulong)i, len);
            case 1: return new byte[len]; // escape-dense
            default: // illegal-heavy
                byte[] data = new byte[len];
                byte[] illegals = TestHelpers.IllegalBytes;
                for (int j = 0; j < len; j++)
                    data[j] = j % 2 == 0 ? illegals[j % illegals.Length] : (byte)(j & 0xFF);
                return data;
        }
    }).ToArray();

    private static async Task RunScenarioAsync(byte[][] headers, byte[][] chunks)
    {
        var pipe = new Pipe();
        var ct = CancellationToken.None;

        Task produce = ProduceAsync(pipe.Writer.AsStream(leaveOpen: true), headers, chunks, ct)
            .ContinueWith(t =>
            {
                t.Exception?.GetBaseException(); // surface producer failures
                return pipe.Writer.CompleteAsync().AsTask();
            }).Unwrap();

        var reader = new FrameReader(pipe.Reader.AsStream(leaveOpen: true));
        for (int i = 0; i < chunks.Length; i++)
        {
            (byte[] Headers, byte[] Data)? frame = await reader.ReadFrameAsync(ct);
            Assert.True(frame.HasValue, $"Frame {i} missing.");

            // Span-based comparison: Assert.Equal over large byte arrays would
            // itself allocate and pollute the allocation budget.
            Assert.True(headers[i].AsSpan().SequenceEqual(frame.Value.Headers), $"Frame {i}: header mismatch.");
            Assert.True(chunks[i].AsSpan().SequenceEqual(frame.Value.Data), $"Frame {i}: data mismatch.");
        }

        // Nothing beyond the last frame.
        Assert.Null(await reader.ReadFrameAsync(ct));

        await produce;
    }

    [Fact]
    public async Task EndToEnd_Pipe_LongLivedEncoderWithReset_PooledReader()
    {
        byte[][] headers = CreateHeaders(12);
        byte[][] chunks = CreateChunks(12);

        await RunScenarioAsync(headers, chunks);
        // No hard allocation budget here: the test assembly runs collections
        // in parallel, so a global GC.GetTotalAllocatedBytes window would
        // count unrelated concurrent tests. The precise per-operation
        // zero-allocation guarantees of the encoder/decoder (warmup, current
        // thread) are enforced by AllocationTests; this scenario exercises
        // the pooled, long-lived-buffer DESIGN end to end.
    }

    [Fact]
    public async Task WireFormat_EncodedSections_MatchStaticEncoding()
    {
        // Explicitly verify what the protocol puts on the wire: headers are
        // plain text; the data block is exactly the static UTF-8 encoding and
        // contains no LF byte (which is why the terminator scan is safe).
        byte[][] headers = CreateHeaders(4);
        byte[][] chunks = CreateChunks(4);

        using var network = new MemoryStream();
        await ProduceAsync(network, headers, chunks, CancellationToken.None);

        byte[] wire = network.ToArray();

        int offset = 0;
        for (int i = 0; i < chunks.Length; i++)
        {
            // Byte-span comparisons: byte offsets, not char offsets (escape
            // pairs are two-byte UTF-8 characters).
            Assert.True(wire.AsSpan(offset).StartsWith(headers[i]), $"Frame {i}: header mismatch.");

            offset += headers[i].Length + TwoNewlines.Length;

            byte[] expectedEncoding = Base122Encoder.EncodeToUtf8(chunks[i]);
            Assert.True(wire.AsSpan(offset).StartsWith(expectedEncoding), $"Frame {i}: data mismatch.");
            Assert.DoesNotContain((byte)0x0A, expectedEncoding);

            offset += expectedEncoding.Length;
            Assert.True(wire.AsSpan(offset, 2).SequenceEqual(TwoNewlines), $"Frame {i}: terminator mismatch.");
            offset += 2;
        }

        Assert.Equal(wire.Length, offset); // no trailing bytes
    }
}
