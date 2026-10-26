using System.Buffers.Binary;

namespace Kanawanagasaki.Base122;

/// <summary>
/// A streaming Base122 decoder realized as a read-only <see cref="Stream"/>:
/// it reads Base122 text (UTF-8) from the input stream supplied at
/// construction and exposes the decoded binary data through
/// <see cref="Read(System.Span{byte})"/> overloads.
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

    private readonly Stream _input;
    private readonly byte[] _buffer;
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
    public Base122Decoder(Stream input) : this(input, leaveOpen: false) { }

    /// <summary>Creates a decoder that reads Base122 text (UTF-8) from <paramref name="input"/>.</summary>
    /// <param name="input">The stream supplying the encoded text.</param>
    /// <param name="leaveOpen">When true, the input stream outlives this decoder.</param>
    public Base122Decoder(Stream input, bool leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(input);
        _input = input;
        _leaveOpen = leaveOpen;
        _buffer = new byte[DefaultBufferSize];
    }

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
}
