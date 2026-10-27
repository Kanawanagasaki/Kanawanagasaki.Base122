using System.Buffers;
using System.Buffers.Binary;

namespace Kanawanagasaki.Base122;

/// <summary>
/// Base122 decoding as a read-only <see cref="Stream"/>: it reads Base122
/// text (UTF-8) from the input stream supplied at construction and exposes
/// the decoded binary data through <see cref="Read(System.Span{byte})"/>
/// overloads.
/// <para>
/// The class also hosts the static in-memory decode API
/// (<see cref="Decode(string)"/>, <see cref="DecodeFromUtf8(ReadOnlySpan{byte})"/>,
/// including Stream, <see cref="IBufferWriter{Byte}"/> and async sources).
/// </para>
/// </summary>
/// <remarks>
/// Because the class is a <see cref="Stream"/>, it can pipe into any other
/// stream (e.g. <c>decoder.CopyTo(output)</c> or
/// <c>await decoder.CopyToAsync(output)</c>), be wrapped by readers such as
/// <c>StreamReader</c>, and be composed with further stream adapters.
/// <para>
/// The decoder is incremental: escape sequences and groups that straddle
/// internal buffer boundaries are handled transparently, and decoded bytes
/// that do not yet fill the caller's buffer are carried over to the next read.
/// </para>
/// Invalid input (a lead byte or continuation byte that cannot occur in
/// Base122 text, or an escape sequence truncated by the end of the stream)
/// throws <see cref="InvalidDataException"/>. Instances are not thread-safe.
/// </remarks>
public sealed class Base122Decoder : Stream
{
    private const int DefaultBufferSize = 8192;
    private const int MaxStackallocBytes = 512;

    /// <summary>Work buffers above ArrayPool's largest bucket (2^20) would not be pooled — those inputs go through the streaming path instead.</summary>
    private const int MaxPooledRentBytes = 1 << 20;

    /// <summary>Staging buffer for the streaming path: large enough that base-Stream per-write task overhead is negligible, small enough to stay bounded.</summary>
    private const int LargePathStagingBytes = 64 * 1024;

    private readonly Stream _input;
    private byte[] _buffer = null!;
    private readonly bool _leaveOpen;

    private int _bufferPos;
    private int _bufferLen;
    private ulong _acc;
    private int _accBits;
    private bool _hasPendingLead;
    private int _pendingCode;
    private bool _endOfStream;
    private bool _disposed;

    /// <summary>Creates a decoder that reads Base122 text (UTF-8) from <paramref name="input"/>.</summary>
    /// <remarks>The input stream is disposed together with the decoder.</remarks>
    public Base122Decoder(Stream input) : this(input, null, leaveOpen: false) { }

    /// <summary>Creates a decoder that reads Base122 text (UTF-8) from <paramref name="input"/>.</summary>
    /// <param name="input">The stream supplying the encoded text.</param>
    /// <param name="leaveOpen">When true, the input stream outlives this decoder.</param>
    public Base122Decoder(Stream input, bool leaveOpen) : this(input, null, leaveOpen) { }

    /// <summary>Creates a decoder that reads Base122 text (UTF-8) from <paramref name="input"/>.</summary>
    /// <param name="input">The stream supplying the encoded text.</param>
    /// <param name="buffer">
    /// A caller-provided buffer used to stage input read from
    /// <paramref name="input"/> before it is decoded — pass e.g.
    /// <c>ArrayPool&lt;byte&gt;.Shared.Rent(8192)</c> and return it after
    /// disposal to avoid library-side allocations entirely. When null, an
    /// 8192-byte buffer is allocated on first use and reused for the
    /// lifetime of the decoder.
    /// </param>
    /// <param name="leaveOpen">When true, the input stream outlives this decoder.</param>
    public Base122Decoder(Stream input, byte[]? buffer, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (buffer is { Length: 0 })
            throw new ArgumentOutOfRangeException(nameof(buffer), "The buffer must not be empty.");
        _input = input;
        _buffer = buffer!;
        _leaveOpen = leaveOpen;
    }

    /// <summary>Allocates the staging buffer on first use (unless provided by the caller).</summary>
    private void EnsureBuffer() => _buffer ??= new byte[DefaultBufferSize];

    #region Stream contract: read-only, non-seekable

    /// <summary>The decoder exposes decoded bytes through <see cref="Read(System.Span{byte})"/>.</summary>
    public override bool CanRead => true;

    /// <summary>The decoder is a read-only stream.</summary>
    public override bool CanSeek => false;

    /// <summary>The decoder is a read-only stream.</summary>
    public override bool CanWrite => false;

    /// <summary>Not supported: a non-seekable decoding source has no length.</summary>
    public override long Length => throw new NotSupportedException("The decoder is a non-seekable read-only stream.");

    /// <summary>Not supported: the decoder is a non-seekable stream.</summary>
    public override long Position
    {
        get => throw new NotSupportedException("The decoder is a non-seekable read-only stream.");
        set => throw new NotSupportedException("The decoder is a non-seekable read-only stream.");
    }

    /// <summary>Not supported: the decoder is a non-seekable stream.</summary>
    public override long Seek(long offset, SeekOrigin origin) =>
        throw new NotSupportedException("The decoder is a non-seekable read-only stream.");

    /// <summary>Not supported: the decoder is a non-seekable stream.</summary>
    public override void SetLength(long value) =>
        throw new NotSupportedException("The decoder is a non-seekable read-only stream.");

    /// <summary>Not supported: the decoder is a read-only stream.</summary>
    public override void Write(byte[] buffer, int offset, int count) =>
        throw new NotSupportedException("The decoder is a read-only stream; it cannot be written to.");

    /// <summary>Not supported: the decoder is a read-only stream.</summary>
    public override void Write(ReadOnlySpan<byte> buffer) =>
        throw new NotSupportedException("The decoder is a read-only stream; it cannot be written to.");

    /// <summary>Not supported: the decoder is a read-only stream.</summary>
    public override void WriteByte(byte value) =>
        throw new NotSupportedException("The decoder is a read-only stream; it cannot be written to.");

    /// <summary>No-op: a read-only decoder holds no data that needs flushing.</summary>
    public override void Flush()
    {
        ThrowIfDisposed();
    }

    /// <summary>No-op: a read-only decoder holds no data that needs flushing.</summary>
    public override Task FlushAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        return Task.CompletedTask;
    }

    #endregion

    #region Lifetime

    /// <summary>Disposes the underlying stream unless it was opened with <c>leaveOpen: true</c>.</summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        _disposed = true;
        if (disposing && !_leaveOpen)
            _input.Dispose();
    }

    /// <summary>Disposes the underlying stream unless it was opened with <c>leaveOpen: true</c>.</summary>
    public override async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        if (!_leaveOpen)
            await _input.DisposeAsync().ConfigureAwait(false);
    }

    #endregion

    /// <summary>
    /// Reads decoded bytes into <paramref name="buffer"/>. Returns the number
    /// of bytes decoded, which is zero only when the end of the input stream
    /// has been reached. The call returns as soon as the buffer is full or the
    /// input is exhausted; partial progress is never lost.
    /// </summary>
    public override int Read(Span<byte> buffer)
    {
        ThrowIfDisposed();
        if (buffer.IsEmpty)
            return 0;
        EnsureBuffer();

        int written = 0;

        while (written < buffer.Length)
        {
            if (_hasPendingLead)
            {
                // Consume the continuation byte of a previously seen escape lead.
                int b2 = ReadInputByte();
                if (b2 < 0)
                    throw new InvalidDataException("Unexpected end of input: truncated Base122 escape sequence.");
                if ((b2 & 0xC0) != 0x80)
                    throw new InvalidDataException($"Invalid Base122 continuation byte 0x{b2:X2}.");
                _hasPendingLead = false;

                int code = _pendingCode;
                int next = ((code & 1) << 6) | (b2 & 0x3F);
                if ((code >> 1) < 6)
                    Push14(Base122Core.Illegals[code >> 1], next, buffer, ref written);
                else
                    Push7(next, buffer, ref written);
                continue;
            }

            // Emit bytes already waiting in the accumulator first: a previous
            // read may have ended with a full byte buffered, and the caller's
            // buffer must be filled in order before consuming more input.
            int before = written;
            EmitAccumulated(buffer, ref written);
            if (written > before)
                continue;

            if (_accBits == 0)
            {
                int available = FillBuffer(8);
                if (available == 0)
                    break; // end of stream

                if (available >= 8
                    && buffer.Length - written >= 7
                    && TryDecodeCleanWindow(buffer.Slice(written)))
                {
                    written += 7;
                    continue;
                }
            }

            int b = ReadInputByte();
            if (b < 0)
                break; // leftover accumulator bits are the final zero padding

            if (b < 0x80)
            {
                Push7(b, buffer, ref written);
            }
            else
            {
                int code = Base122Core.LeadDecode[b];
                if (code == 0xFF)
                    throw new InvalidDataException($"Invalid Base122 lead byte 0x{b:X2}.");
                _pendingCode = code;
                _hasPendingLead = true;
            }
        }

        return written;
    }

    /// <summary>Reads decoded bytes into <paramref name="buffer"/>.</summary>
    public override int Read(byte[] buffer, int offset, int count)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if ((uint)offset > (uint)buffer.Length || buffer.Length - offset < count)
            throw new ArgumentOutOfRangeException(nameof(offset), "Offset and count are out of bounds for the array.");

        return Read(buffer.AsSpan(offset, count));
    }

    /// <summary>
    /// Reads decoded bytes into <paramref name="buffer"/>. Returns the number
    /// of bytes decoded, which is zero only when the end of the input stream
    /// has been reached. The only awaited operations are reads from the
    /// underlying stream; partial progress is never lost.
    /// </summary>
    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (cancellationToken.IsCancellationRequested)
            return ValueTask.FromCanceled<int>(cancellationToken);
        if (buffer.IsEmpty)
            return new ValueTask<int>(0);
        EnsureBuffer();
        return ReadAsyncCore(buffer, cancellationToken);
    }

    private async ValueTask<int> ReadAsyncCore(Memory<byte> buffer, CancellationToken cancellationToken)
    {
        int written = 0;

        while (written < buffer.Length)
        {
            if (_hasPendingLead)
            {
                // Consume the continuation byte of a previously seen escape lead.
                int b2 = await ReadInputByteAsync(cancellationToken).ConfigureAwait(false);
                if (b2 < 0)
                    throw new InvalidDataException("Unexpected end of input: truncated Base122 escape sequence.");
                if ((b2 & 0xC0) != 0x80)
                    throw new InvalidDataException($"Invalid Base122 continuation byte 0x{b2:X2}.");
                _hasPendingLead = false;

                int code = _pendingCode;
                int next = ((code & 1) << 6) | (b2 & 0x3F);
                if ((code >> 1) < 6)
                    Push14(Base122Core.Illegals[code >> 1], next, buffer.Span, ref written);
                else
                    Push7(next, buffer.Span, ref written);
                continue;
            }

            int before = written;
            EmitAccumulated(buffer.Span, ref written);
            if (written > before)
                continue;

            if (_accBits == 0)
            {
                int available = await FillBufferAsync(8, cancellationToken).ConfigureAwait(false);
                if (available == 0)
                    break; // end of stream

                if (available >= 8
                    && buffer.Length - written >= 7
                    && TryDecodeCleanWindow(buffer.Span.Slice(written)))
                {
                    written += 7;
                    continue;
                }
            }

            int b = await ReadInputByteAsync(cancellationToken).ConfigureAwait(false);
            if (b < 0)
                break; // leftover accumulator bits are the final zero padding

            if (b < 0x80)
            {
                Push7(b, buffer.Span, ref written);
            }
            else
            {
                int code = Base122Core.LeadDecode[b];
                if (code == 0xFF)
                    throw new InvalidDataException($"Invalid Base122 lead byte 0x{b:X2}.");
                _pendingCode = code;
                _hasPendingLead = true;
            }
        }

        return written;
    }

    /// <summary>Asynchronously reads decoded bytes into <paramref name="buffer"/>.</summary>
    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if ((uint)offset > (uint)buffer.Length || buffer.Length - offset < count)
            throw new ArgumentOutOfRangeException(nameof(offset), "Offset and count are out of bounds for the array.");

        return ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }

    /// <summary>
    /// Reads decoded bytes one at a time, or -1 when the end of the input
    /// stream has been reached.
    /// </summary>
    public override int ReadByte()
    {
        Span<byte> single = stackalloc byte[1];
        int read = Read(single);
        return read == 1 ? single[0] : -1;
    }

    /// <summary>
    /// Decodes a window of eight consecutive clean (ASCII) bytes into seven
    /// output bytes in one shot; returns false when the window contains an
    /// escape byte, leaving the buffered input untouched.
    /// </summary>
    private bool TryDecodeCleanWindow(Span<byte> output)
    {
        ulong chunk = BinaryPrimitives.ReadUInt64LittleEndian(_buffer.AsSpan(_bufferPos, 8));
        if ((chunk & 0x8080808080808080UL) != 0)
            return false;

        ulong bits56 = Base122Core.GatherGroupsToBits56(chunk);

        output[0] = (byte)(bits56 >> 48);
        output[1] = (byte)(bits56 >> 40);
        output[2] = (byte)(bits56 >> 32);
        output[3] = (byte)(bits56 >> 24);
        output[4] = (byte)(bits56 >> 16);
        output[5] = (byte)(bits56 >> 8);
        output[6] = (byte)bits56;

        _bufferPos += 8;
        return true;
    }

    private void Push7(int bits, Span<byte> output, ref int written)
    {
        _acc = (_acc << 7) | (uint)bits;
        _accBits += 7;
        EmitAccumulated(output, ref written);
    }

    private void Push14(int illegalValue, int next, Span<byte> output, ref int written)
    {
        _acc = (_acc << 14) | ((uint)illegalValue << 7) | (uint)next;
        _accBits += 14;
        EmitAccumulated(output, ref written);
    }

    private void EmitAccumulated(Span<byte> output, ref int written)
    {
        while (_accBits >= 8 && written < output.Length)
        {
            _accBits -= 8;
            output[written++] = (byte)(_acc >> _accBits);
        }
        _acc &= (1UL << _accBits) - 1;
    }

    /// <summary>
    /// Ensures buffered input is available, compacting and refilling the
    /// buffer with a single read when possible. Returns the number of buffered
    /// bytes; zero means the end of the input stream.
    /// </summary>
    private int FillBuffer(int minBytes)
    {
        int available = _bufferLen - _bufferPos;
        if (available >= minBytes)
            return available;

        if (_bufferPos > 0)
        {
            Array.Copy(_buffer, _bufferPos, _buffer, 0, available);
            _bufferPos = 0;
            _bufferLen = available;
        }

        if (!_endOfStream && _bufferLen < minBytes)
        {
            int read = _input.Read(_buffer.AsSpan(_bufferLen));
            if (read <= 0)
            {
                _endOfStream = true;
            }
            else
            {
                _bufferLen += read;
            }
        }

        return _bufferLen - _bufferPos;
    }

    private int ReadInputByte()
    {
        if (FillBuffer(1) == 0)
            return -1;
        return _buffer[_bufferPos++];
    }

    /// <summary>
    /// Asynchronously ensures buffered input is available, compacting and
    /// refilling the buffer with a single read when possible. Returns the
    /// number of buffered bytes; zero means the end of the input stream.
    /// </summary>
    private async ValueTask<int> FillBufferAsync(int minBytes, CancellationToken cancellationToken)
    {
        int available = _bufferLen - _bufferPos;
        if (available >= minBytes)
            return available;

        if (_bufferPos > 0)
        {
            Array.Copy(_buffer, _bufferPos, _buffer, 0, available);
            _bufferPos = 0;
            _bufferLen = available;
        }

        if (!_endOfStream && _bufferLen < minBytes)
        {
            int read = await _input.ReadAsync(_buffer.AsMemory(_bufferLen), cancellationToken).ConfigureAwait(false);
            if (read <= 0)
            {
                _endOfStream = true;
            }
            else
            {
                _bufferLen += read;
            }
        }

        return _bufferLen - _bufferPos;
    }

    private async ValueTask<int> ReadInputByteAsync(CancellationToken cancellationToken)
    {
        if (await FillBufferAsync(1, cancellationToken).ConfigureAwait(false) == 0)
            return -1;
        return _buffer[_bufferPos++];
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(Base122Decoder));
    }

    #region Static in-memory decode API

    /// <summary>
    /// Decodes a Base122 string back into its original binary form.
    /// </summary>
    /// <exception cref="FormatException">
    /// The input contains a character that cannot occur in Base122 text.
    /// </exception>
    public static byte[] Decode(ReadOnlySpan<char> encoded)
    {
        if (encoded.IsEmpty)
            return [];

        int maxByteCount = GetMaxDecodedLength(encoded.Length);

        if (maxByteCount <= MaxStackallocBytes)
        {
            Span<byte> buffer = stackalloc byte[MaxStackallocBytes];
            if (!Base122Core.Decode(encoded, buffer, out int written, out int errorOffset))
                throw InvalidInput(encoded, errorOffset);
            return buffer[..written].ToArray();
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(maxByteCount);
        try
        {
            if (!Base122Core.Decode(encoded, rented, out int written, out int errorOffset))
                throw InvalidInput(encoded, errorOffset);
            return rented.AsSpan(0, written).ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Decodes a Base122 string back into its original binary form.
    /// </summary>
    /// <exception cref="FormatException">
    /// The input contains a character that cannot occur in Base122 text.
    /// </exception>
    public static byte[] Decode(string encoded)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        return Decode(encoded.AsSpan());
    }

    /// <summary>
    /// Decodes a Base122 text into a caller-provided buffer.
    /// <paramref name="destination"/> must be at least
    /// <see cref="GetMaxDecodedLength"/> bytes long.
    /// </summary>
    /// <param name="encoded">The Base122 text to decode.</param>
    /// <param name="destination">The buffer receiving the decoded bytes.</param>
    /// <param name="bytesWritten">The exact number of decoded bytes on success.</param>
    /// <returns>
    /// False when <paramref name="destination"/> is too small or when the input
    /// contains a character that cannot occur in Base122 text.
    /// </returns>
    public static bool TryDecode(ReadOnlySpan<char> encoded, Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < GetMaxDecodedLength(encoded.Length))
        {
            bytesWritten = 0;
            return false;
        }

        return Base122Core.Decode(encoded, destination, out bytesWritten, out _);
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of a Base122 text back into its original
    /// binary form — the byte-level counterpart of
    /// <see cref="Decode(ReadOnlySpan{char})"/>, without materializing any
    /// string or character data.
    /// </summary>
    /// <exception cref="FormatException">
    /// The input contains a byte that cannot occur in Base122 UTF-8 text (an
    /// invalid lead byte), an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </exception>
    public static byte[] DecodeFromUtf8(byte[] encoded)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        return DecodeFromUtf8(encoded.AsSpan());
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of a Base122 text back into its original
    /// binary form — the byte-level counterpart of
    /// <see cref="Decode(ReadOnlySpan{char})"/>, without materializing any
    /// string or character data.
    /// </summary>
    /// <remarks>
    /// The exact decoded length is computed in O(1) (every input byte carries
    /// exactly 7 bits, except a single ending shortened escape), so the result
    /// array is the only allocation.
    /// </remarks>
    /// <exception cref="FormatException">
    /// The input contains a byte that cannot occur in Base122 UTF-8 text (an
    /// invalid lead byte), an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </exception>
    public static byte[] DecodeFromUtf8(ReadOnlySpan<byte> encoded)
    {
        if (encoded.IsEmpty)
            return [];

        byte[] result = new byte[Base122Core.GetDecodedFromUtf8Length(encoded)];
        if (!Base122Core.DecodeFromUtf8(encoded, result, out int written, out int errorOffset))
            throw InvalidUtf8Input(encoded, errorOffset);

        if (written != result.Length)
        {
            // Non-canonical (but decodable) input decoded to fewer bytes.
            return result[..written].ToArray();
        }
        return result;
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of a Base122 text into
    /// <paramref name="bufferWriter"/>.
    /// </summary>
    /// <param name="encoded">The UTF-8 bytes of the Base122 text to decode.</param>
    /// <param name="bufferWriter">The writer receiving the decoded bytes.</param>
    /// <returns>The number of decoded bytes written to the writer.</returns>
    /// <exception cref="FormatException">
    /// The input contains a byte that cannot occur in Base122 UTF-8 text (an
    /// invalid lead byte), an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </exception>
    public static int DecodeFromUtf8(ReadOnlySpan<byte> encoded, IBufferWriter<byte> bufferWriter)
    {
        ArgumentNullException.ThrowIfNull(bufferWriter);

        Span<byte> span = bufferWriter.GetSpan(GetMaxDecodedFromUtf8Length(encoded.Length));
        if (!Base122Core.DecodeFromUtf8(encoded, span, out int written, out int errorOffset))
            throw InvalidUtf8Input(encoded, errorOffset);
        bufferWriter.Advance(written);
        return written;
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of a Base122 text and writes the decoded
    /// bytes to <paramref name="destination"/>.
    /// </summary>
    /// <param name="encoded">The UTF-8 bytes of the Base122 text to decode.</param>
    /// <param name="destination">The stream receiving the decoded bytes.</param>
    /// <returns>The number of decoded bytes written to the stream.</returns>
    /// <exception cref="FormatException">
    /// The input contains a byte that cannot occur in Base122 UTF-8 text (an
    /// invalid lead byte), an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </exception>
    /// <remarks>
    /// Decodes into a pooled buffer sized
    /// <see cref="GetMaxDecodedFromUtf8Length"/> and writes once — one pooled
    /// rent, no steady-state allocation. (For inputs above ~1.2 MB the rent
    /// exceeds ArrayPool's largest bucket and is a direct allocation; use the
    /// async overload or the instance decoder to stream such inputs with a
    /// bounded working set — a span cannot be captured by the streaming
    /// decoder's input stream.)
    /// </remarks>
    public static int DecodeFromUtf8(ReadOnlySpan<byte> encoded, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite)
            throw new NotSupportedException("The destination stream must be writable.");
        if (encoded.IsEmpty)
            return 0;

        byte[] rented = ArrayPool<byte>.Shared.Rent(GetMaxDecodedFromUtf8Length(encoded.Length));
        try
        {
            if (!Base122Core.DecodeFromUtf8(encoded, rented, out int written, out int errorOffset))
                throw InvalidUtf8Input(encoded, errorOffset);
            destination.Write(rented, 0, written);
            return written;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Asynchronously decodes the UTF-8 byte form of a Base122 text and
    /// writes the decoded bytes to <paramref name="destination"/> using
    /// asynchronous I/O.
    /// </summary>
    /// <param name="encoded">The UTF-8 bytes of the Base122 text to decode.</param>
    /// <param name="destination">The stream receiving the decoded bytes.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>The number of decoded bytes written to the stream.</returns>
    /// <exception cref="FormatException">
    /// The input contains a byte that cannot occur in Base122 UTF-8 text (an
    /// invalid lead byte), an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </exception>
    /// <remarks>
    /// Inputs whose decoded output fits in 1 MB are decoded into a pooled
    /// buffer and written with a single async write — no steady-state
    /// allocation. Larger inputs are streamed through a
    /// <see cref="Base122Decoder"/> with a pooled 8 KB staging buffer, keeping
    /// the working set bounded.
    /// </remarks>
    public static async Task<int> DecodeFromUtf8Async(ReadOnlyMemory<byte> encoded, Stream destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite)
            throw new NotSupportedException("The destination stream must be writable.");
        if (encoded.IsEmpty)
            return 0;

        int max = GetMaxDecodedFromUtf8Length(encoded.Length);
        if (max <= MaxPooledRentBytes)
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(max);
            try
            {
                if (!Base122Core.DecodeFromUtf8(encoded.Span, rented, out int written, out int errorOffset))
                    throw InvalidUtf8Input(encoded.Span, errorOffset);
                await destination.WriteAsync(rented.AsMemory(0, written), cancellationToken).ConfigureAwait(false);
                return written;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        byte[] inputStaging = ArrayPool<byte>.Shared.Rent(DefaultBufferSize);
        byte[] outStaging = ArrayPool<byte>.Shared.Rent(LargePathStagingBytes);
        try
        {
            await using var decoder = new Base122Decoder(new MemoryInputStream(encoded), inputStaging, leaveOpen: true);
            long total = 0;
            int read;
            while ((read = await decoder.ReadAsync(outStaging, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await destination.WriteAsync(outStaging.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                total += read;
            }
            return (int)total;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(inputStaging);
            ArrayPool<byte>.Shared.Return(outStaging);
        }
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of a Base122 text into a caller-provided buffer.
    /// <paramref name="destination"/> must be at least
    /// <see cref="GetMaxDecodedFromUtf8Length"/> bytes long.
    /// </summary>
    /// <param name="encoded">The UTF-8 bytes of the Base122 text to decode.</param>
    /// <param name="destination">The buffer receiving the decoded bytes.</param>
    /// <param name="bytesWritten">The exact number of decoded bytes on success.</param>
    /// <returns>
    /// False when <paramref name="destination"/> is too small or when the input
    /// contains a byte that cannot occur in Base122 UTF-8 text.
    /// </returns>
    public static bool TryDecodeFromUtf8(ReadOnlySpan<byte> encoded, Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < GetMaxDecodedFromUtf8Length(encoded.Length))
        {
            bytesWritten = 0;
            return false;
        }

        return Base122Core.DecodeFromUtf8(encoded, destination, out bytesWritten, out _);
    }

    /// <summary>
    /// Returns the maximum number of bytes decoding
    /// <paramref name="byteCount"/> bytes of Base122 UTF-8 text can ever produce.
    /// </summary>
    /// <remarks>
    /// Every input byte carries exactly 7 bits (an escape pair carries 14 bits
    /// over two bytes), so the bound is 7*byteCount/8.
    /// </remarks>
    public static int GetMaxDecodedFromUtf8Length(int byteCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(byteCount);
        return (int)(7L * byteCount / 8);
    }

    /// <summary>
    /// Returns the exact number of bytes <see cref="DecodeFromUtf8(ReadOnlySpan{byte})"/>
    /// produces for a canonical encoding, computed in O(1).
    /// </summary>
    /// <remarks>
    /// For non-canonical (but decodable) input the result may be larger than
    /// the actual decoded length; it is still a safe upper bound. Use
    /// <see cref="TryDecodeFromUtf8"/> to decode without allocating.
    /// </remarks>
    public static int GetDecodedFromUtf8Length(ReadOnlySpan<byte> encoded) =>
        Base122Core.GetDecodedFromUtf8Length(encoded);

    /// <summary>
    /// Returns the maximum number of bytes decoding
    /// <paramref name="charCount"/> characters of Base122 text can ever produce.
    /// </summary>
    /// <remarks>
    /// Every character carries at most 14 bits (an escape character holds an
    /// illegal group plus the following group in a single char), so the bound
    /// is 7*charCount/4 bytes.
    /// </remarks>
    public static int GetMaxDecodedLength(int charCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(charCount);
        return (int)(7L * charCount / 4);
    }

    private static FormatException InvalidUtf8Input(ReadOnlySpan<byte> encoded, int errorOffset)
    {
        byte b = errorOffset < encoded.Length ? encoded[errorOffset] : (byte)0;
        return new FormatException($"Invalid Base122 input: invalid byte 0x{b:X2} at index {errorOffset}.");
    }

    /// <summary>
    /// A forward-only read-only stream over <see cref="ReadOnlyMemory{Byte}"/>
    /// (internal: lets the streaming decoder pump over in-memory input without
    /// copying it into a MemoryStream first).
    /// </summary>
    private sealed class MemoryInputStream(ReadOnlyMemory<byte> memory) : Stream
    {
        private int _pos;

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            // Complete synchronously: the source is already in memory, and the
            // base fallback would suspend (allocating continuations) for no reason.
            if (cancellationToken.IsCancellationRequested)
                return ValueTask.FromCanceled<int>(cancellationToken);
            return new ValueTask<int>(Read(buffer.Span));
        }

        public override int Read(Span<byte> buffer)
        {
            int take = Math.Min(buffer.Length, memory.Length - _pos);
            memory.Span.Slice(_pos, take).CopyTo(buffer);
            _pos += take;
            return take;
        }

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static FormatException InvalidInput(ReadOnlySpan<char> encoded, int errorOffset)
    {
        uint c = errorOffset < encoded.Length ? (uint)encoded[errorOffset] : 0;
        return new FormatException($"Invalid Base122 input: unexpected character U+{c:X4} at index {errorOffset}.");
    }

    #endregion
}
