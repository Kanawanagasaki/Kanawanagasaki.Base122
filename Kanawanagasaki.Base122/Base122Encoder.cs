using System.Buffers.Binary;

namespace Kanawanagasaki.Base122;

/// <summary>
/// A streaming Base122 encoder realized as a write-only <see cref="Stream"/>:
/// raw bytes written to this stream are encoded as Base122 text (UTF-8) into
/// the output stream supplied at construction.
/// </summary>
/// <remarks>
/// Because the class is a <see cref="Stream"/>, other streams can pipe into
/// it (e.g. <c>source.CopyTo(encoder)</c> or <c>await source.CopyToAsync(encoder)</c>).
/// <para>
/// The encoded bit stream may only be finalized once all input has been
/// written: the final (possibly zero-padded) group and any trailing escape are
/// emitted by <see cref="Complete"/> or <see cref="CompleteAsync"/> — which
/// <see cref="Dispose(bool)"/> and <see cref="DisposeAsync"/> call implicitly.
/// </para>
/// Instances are not thread-safe.
/// </remarks>
public sealed class Base122Encoder : Stream
{
    private const int DefaultBufferSize = 8192;

    private readonly Stream _output;
    private readonly byte[] _buffer;
    private readonly bool _leaveOpen;

    private int _buffered;
    private ulong _acc;
    private int _accBits;
    private bool _hasPendingIllegal;
    private int _pendingIllegal;
    private bool _completed;
    private bool _disposed;

    /// <summary>Creates an encoder that writes Base122 text (UTF-8) to <paramref name="output"/>.</summary>
    /// <remarks>The output stream is disposed together with the encoder.</remarks>
    public Base122Encoder(Stream output) : this(output, leaveOpen: false) { }

    /// <summary>Creates an encoder that writes Base122 text (UTF-8) to <paramref name="output"/>.</summary>
    /// <param name="output">The stream receiving the encoded text.</param>
    /// <param name="leaveOpen">When true, the output stream outlives this encoder.</param>
    public Base122Encoder(Stream output, bool leaveOpen)
    {
        ArgumentNullException.ThrowIfNull(output);
        _output = output;
        _leaveOpen = leaveOpen;
        _buffer = new byte[DefaultBufferSize];
    }

    #region Stream contract: write-only, non-seekable

    /// <summary>The encoder is a write-only stream.</summary>
    public override bool CanRead => false;

    /// <summary>The encoder is a write-only stream.</summary>
    public override bool CanSeek => false;

    /// <summary>The encoder accepts raw bytes via <see cref="Write(ReadOnlySpan{byte})"/>.</summary>
    public override bool CanWrite => true;

    /// <summary>Not supported: a non-seekable encoding sink has no length.</summary>
    public override long Length => throw new NotSupportedException("The encoder is a non-seekable write-only stream.");

    /// <summary>Not supported: the encoder is a non-seekable stream.</summary>
    public override long Position
    {
        get => throw new NotSupportedException("The encoder is a non-seekable write-only stream.");
        set => throw new NotSupportedException("The encoder is a non-seekable write-only stream.");
    }

    /// <summary>Not supported: the encoder is a non-seekable stream.</summary>
    public override long Seek(long offset, SeekOrigin origin) =>
        throw new NotSupportedException("The encoder is a non-seekable write-only stream.");

    /// <summary>Not supported: the encoder is a non-seekable stream.</summary>
    public override void SetLength(long value) =>
        throw new NotSupportedException("The encoder is a non-seekable write-only stream.");

    /// <summary>Not supported: the encoder is a write-only stream.</summary>
    public override int Read(byte[] buffer, int offset, int count) =>
        throw new NotSupportedException("The encoder is a write-only stream; it cannot be read from.");

    /// <summary>Not supported: the encoder is a write-only stream.</summary>
    public override int Read(Span<byte> buffer) =>
        throw new NotSupportedException("The encoder is a write-only stream; it cannot be read from.");

    /// <summary>
    /// Pushes any buffered encoded output to the underlying stream and flushes
    /// it. This does not finalize the encoding; call <see cref="Complete"/>
    /// after the last write to emit the end of the bit stream.
    /// </summary>
    public override void Flush()
    {
        ThrowIfDisposed();
        FlushBuffer();
        _output.Flush();
    }

    /// <summary>
    /// Asynchronously pushes any buffered encoded output to the underlying
    /// stream and flushes it. This does not finalize the encoding; call
    /// <see cref="CompleteAsync"/> after the last write.
    /// </summary>
    public override async Task FlushAsync(CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        await FlushBufferAsync(cancellationToken).ConfigureAwait(false);
        await _output.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    #endregion

    #region Lifetime

    /// <summary>
    /// Finalizes the encoded bit stream and disposes the underlying stream
    /// unless it was opened with <c>leaveOpen: true</c>.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (!disposing)
        {
            _disposed = true;
            return;
        }

        try
        {
            if (!_completed)
                Complete();
        }
        finally
        {
            _disposed = true;
            if (!_leaveOpen)
                _output.Dispose();
        }
    }

    /// <summary>
    /// Finalizes the encoded bit stream and disposes the underlying stream
    /// unless it was opened with <c>leaveOpen: true</c>.
    /// </summary>
    public override async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        try
        {
            if (!_completed)
                await CompleteAsync().ConfigureAwait(false);
        }
        finally
        {
            _disposed = true;
            if (!_leaveOpen)
                await _output.DisposeAsync().ConfigureAwait(false);
        }
    }

    #endregion

    /// <summary>
    /// Encodes <paramref name="data"/> and writes the result to the output stream.
    /// Data may be written in arbitrarily sized chunks; the bit stream is kept
    /// continuous across calls.
    /// </summary>
    public override void Write(ReadOnlySpan<byte> data)
    {
        ThrowIfWritable();

        int i = 0;

        // Fast path: byte-aligned escape-free runs, 7 bytes -> 8 output bytes,
        // processed with 64-bit SWAR exactly like the in-memory kernel.
        while (_accBits == 0 && !_hasPendingIllegal && i + 8 <= data.Length)
        {
            ulong groups = Base122Core.GroupsLE(BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(i, 8)));
            if (Base122Core.IllegalHighMask(groups) != 0)
                break;

            if (_buffered + 8 > _buffer.Length)
                FlushBuffer();

            _buffer[_buffered] = (byte)groups;
            _buffer[_buffered + 1] = (byte)(groups >> 8);
            _buffer[_buffered + 2] = (byte)(groups >> 16);
            _buffer[_buffered + 3] = (byte)(groups >> 24);
            _buffer[_buffered + 4] = (byte)(groups >> 32);
            _buffer[_buffered + 5] = (byte)(groups >> 40);
            _buffer[_buffered + 6] = (byte)(groups >> 48);
            _buffer[_buffered + 7] = (byte)(groups >> 56);
            _buffered += 8;
            i += 7;
        }

        // General path: pull 7-bit groups from the bit accumulator.
        while (true)
        {
            while (_accBits >= 7)
            {
                _accBits -= 7;
                int group = (int)((_acc >> _accBits) & 0x7F);
                _acc &= (1UL << _accBits) - 1;
                ProcessGroup(group);
            }

            if (i >= data.Length)
                break;

            _acc = (_acc << 8) | data[i++];
            _accBits += 8;
        }
    }

    /// <summary>Encodes <paramref name="data"/> and writes the result to the output stream.</summary>
    public void Write(byte[] data) => Write(data.AsSpan());

    /// <summary>Encodes <paramref name="data"/> and writes the result to the output stream.</summary>
    public override void Write(byte[] data, int offset, int count)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if ((uint)offset > (uint)data.Length || data.Length - offset < count)
            throw new ArgumentOutOfRangeException(nameof(offset), "Offset and count are out of bounds for the array.");

        Write(data.AsSpan(offset, count));
    }

    /// <summary>Encodes a single raw byte (no allocation, unlike the base default).</summary>
    public override void WriteByte(byte value) => Write(new ReadOnlySpan<byte>(in value));

    /// <summary>
    /// Asynchronously encodes <paramref name="buffer"/> and writes the result
    /// to the output stream. The only awaited operations are flushes of the
    /// internal buffer to the underlying stream; chunks may be of any size and
    /// the bit stream is kept continuous across calls.
    /// </summary>
    public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        ThrowIfWritable();
        if (cancellationToken.IsCancellationRequested)
            return ValueTask.FromCanceled(cancellationToken);
        return WriteAsyncCore(buffer, cancellationToken);
    }

    /// <summary>Asynchronously encodes <paramref name="buffer"/> and writes the result to the output stream.</summary>
    public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if ((uint)offset > (uint)buffer.Length || buffer.Length - offset < count)
            throw new ArgumentOutOfRangeException(nameof(offset), "Offset and count are out of bounds for the array.");

        ThrowIfWritable();
        if (cancellationToken.IsCancellationRequested)
            return Task.FromCanceled(cancellationToken);
        return WriteAsyncCore(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }

    private async ValueTask WriteAsyncCore(ReadOnlyMemory<byte> data, CancellationToken cancellationToken)
    {
        int i = 0;

        // Fast path: byte-aligned escape-free runs, 7 bytes -> 8 output bytes.
        while (_accBits == 0 && !_hasPendingIllegal && i + 8 <= data.Length)
        {
            ulong groups = Base122Core.GroupsLE(BinaryPrimitives.ReadUInt64LittleEndian(data.Span.Slice(i, 8)));
            if (Base122Core.IllegalHighMask(groups) != 0)
                break;

            if (_buffered + 8 > _buffer.Length)
                await FlushBufferAsync(cancellationToken).ConfigureAwait(false);

            _buffer[_buffered] = (byte)groups;
            _buffer[_buffered + 1] = (byte)(groups >> 8);
            _buffer[_buffered + 2] = (byte)(groups >> 16);
            _buffer[_buffered + 3] = (byte)(groups >> 24);
            _buffer[_buffered + 4] = (byte)(groups >> 32);
            _buffer[_buffered + 5] = (byte)(groups >> 40);
            _buffer[_buffered + 6] = (byte)(groups >> 48);
            _buffer[_buffered + 7] = (byte)(groups >> 56);
            _buffered += 8;
            i += 7;
        }

        // General path: pull 7-bit groups from the bit accumulator.
        while (true)
        {
            while (_accBits >= 7)
            {
                _accBits -= 7;
                int group = (int)((_acc >> _accBits) & 0x7F);
                _acc &= (1UL << _accBits) - 1;
                await ProcessGroupAsync(group, cancellationToken).ConfigureAwait(false);
            }

            if (i >= data.Length)
                break;

            _acc = (_acc << 8) | data.Span[i++];
            _accBits += 8;
        }
    }

    /// <summary>
    /// Finalizes the encoded bit stream: emits the final (possibly zero-padded)
    /// group and any trailing escape, then flushes. No further data may be
    /// written afterwards. <see cref="Dispose(bool)"/> calls this automatically.
    /// </summary>
    public void Complete()
    {
        ThrowIfDisposed();
        if (_completed)
            return;
        _completed = true;

        // Drain any remaining full groups (Write normally leaves fewer than
        // 7 bits, so this is defensive).
        while (_accBits >= 7)
        {
            _accBits -= 7;
            int group = (int)((_acc >> _accBits) & 0x7F);
            _acc &= (1UL << _accBits) - 1;
            ProcessGroup(group);
        }

        if (_hasPendingIllegal)
        {
            if (_accBits > 0)
            {
                // The pending illegal group pairs with the final zero-padded group.
                int next = (int)((_acc << (7 - _accBits)) & 0x7F);
                WriteEscapePair(_pendingIllegal, next);
            }
            else
            {
                // Nothing follows: emit the shortened escape.
                WriteShortened(_pendingIllegal);
            }
            _hasPendingIllegal = false;
        }
        else if (_accBits > 0)
        {
            int group = (int)((_acc << (7 - _accBits)) & 0x7F);
            if (Base122Core.Class[group] != 0)
                WriteShortened(group);
            else
                EmitByte((byte)group);
        }

        _acc = 0;
        _accBits = 0;

        FlushBuffer();
        _output.Flush();
    }

    /// <summary>
    /// Asynchronously finalizes the encoded bit stream: emits the final
    /// (possibly zero-padded) group and any trailing escape, then flushes.
    /// No further data may be written afterwards.
    /// <see cref="DisposeAsync"/> calls this automatically.
    /// </summary>
    public async Task CompleteAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (_completed)
            return;
        _completed = true;

        // Drain any remaining full groups (Write normally leaves fewer than
        // 7 bits, so this is defensive).
        while (_accBits >= 7)
        {
            _accBits -= 7;
            int group = (int)((_acc >> _accBits) & 0x7F);
            _acc &= (1UL << _accBits) - 1;
            await ProcessGroupAsync(group, cancellationToken).ConfigureAwait(false);
        }

        if (_hasPendingIllegal)
        {
            if (_accBits > 0)
            {
                int next = (int)((_acc << (7 - _accBits)) & 0x7F);
                await WriteEscapePairAsync(_pendingIllegal, next, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await WriteShortenedAsync(_pendingIllegal, cancellationToken).ConfigureAwait(false);
            }
            _hasPendingIllegal = false;
        }
        else if (_accBits > 0)
        {
            int group = (int)((_acc << (7 - _accBits)) & 0x7F);
            if (Base122Core.Class[group] != 0)
                await WriteShortenedAsync(group, cancellationToken).ConfigureAwait(false);
            else
                await WriteByteAsync((byte)group, cancellationToken).ConfigureAwait(false);
        }

        _acc = 0;
        _accBits = 0;

        await FlushBufferAsync(cancellationToken).ConfigureAwait(false);
        await _output.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    #region Sync emission (private)

    private void ProcessGroup(int group)
    {
        if (_hasPendingIllegal)
        {
            // One escape carries the pending illegal group and this group.
            WriteEscapePair(_pendingIllegal, group);
            _hasPendingIllegal = false;
        }
        else if (Base122Core.Class[group] != 0)
        {
            _pendingIllegal = group;
            _hasPendingIllegal = true;
        }
        else
        {
            EmitByte((byte)group);
        }
    }

    private void EmitByte(byte value)
    {
        if (_buffered + 1 > _buffer.Length)
            FlushBuffer();
        _buffer[_buffered++] = value;
    }

    private void WriteEscapePair(int illegalValue, int next)
    {
        ushort pair = Base122Core.EscapePairs[((Base122Core.Class[illegalValue] - 1) << 7) | next];
        WriteTwoBytes((byte)pair, (byte)(pair >> 8));
    }

    private void WriteShortened(int group)
    {
        ushort pair = Base122Core.Shortened[group];
        WriteTwoBytes((byte)pair, (byte)(pair >> 8));
    }

    private void WriteTwoBytes(byte first, byte second)
    {
        if (_buffered + 2 > _buffer.Length)
            FlushBuffer();
        _buffer[_buffered++] = first;
        _buffer[_buffered++] = second;
    }

    private void FlushBuffer()
    {
        if (_buffered > 0)
        {
            _output.Write(_buffer, 0, _buffered);
            _buffered = 0;
        }
    }

    #endregion

    #region Async emission (private)

    // The async emission helpers mirror the sync ones, but complete
    // synchronously (no state machine) whenever the internal buffer has room —
    // only an actual flush to the underlying stream awaits.

    private ValueTask ProcessGroupAsync(int group, CancellationToken cancellationToken)
    {
        if (_hasPendingIllegal)
        {
            _hasPendingIllegal = false;
            return WriteEscapePairAsync(_pendingIllegal, group, cancellationToken);
        }
        if (Base122Core.Class[group] != 0)
        {
            _pendingIllegal = group;
            _hasPendingIllegal = true;
            return ValueTask.CompletedTask;
        }
        return WriteByteAsync((byte)group, cancellationToken);
    }

    private ValueTask WriteByteAsync(byte value, CancellationToken cancellationToken)
    {
        if (_buffered + 1 <= _buffer.Length)
        {
            _buffer[_buffered++] = value;
            return ValueTask.CompletedTask;
        }
        return FlushAndWriteByteAsync(value, cancellationToken);
    }

    private async ValueTask FlushAndWriteByteAsync(byte value, CancellationToken cancellationToken)
    {
        await FlushBufferAsync(cancellationToken).ConfigureAwait(false);
        _buffer[_buffered++] = value;
    }

    private ValueTask WriteEscapePairAsync(int illegalValue, int next, CancellationToken cancellationToken)
    {
        ushort pair = Base122Core.EscapePairs[((Base122Core.Class[illegalValue] - 1) << 7) | next];
        return WriteTwoBytesAsync((byte)pair, (byte)(pair >> 8), cancellationToken);
    }

    private ValueTask WriteShortenedAsync(int group, CancellationToken cancellationToken)
    {
        ushort pair = Base122Core.Shortened[group];
        return WriteTwoBytesAsync((byte)pair, (byte)(pair >> 8), cancellationToken);
    }

    private ValueTask WriteTwoBytesAsync(byte first, byte second, CancellationToken cancellationToken)
    {
        if (_buffered + 2 <= _buffer.Length)
        {
            _buffer[_buffered++] = first;
            _buffer[_buffered++] = second;
            return ValueTask.CompletedTask;
        }
        return FlushAndWriteTwoBytesAsync(first, second, cancellationToken);
    }

    private async ValueTask FlushAndWriteTwoBytesAsync(byte first, byte second, CancellationToken cancellationToken)
    {
        await FlushBufferAsync(cancellationToken).ConfigureAwait(false);
        _buffer[_buffered++] = first;
        _buffer[_buffered++] = second;
    }

    private async ValueTask FlushBufferAsync(CancellationToken cancellationToken)
    {
        if (_buffered > 0)
        {
            await _output.WriteAsync(_buffer.AsMemory(0, _buffered), cancellationToken).ConfigureAwait(false);
            _buffered = 0;
        }
    }

    #endregion

    private void ThrowIfWritable()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(Base122Encoder));
        if (_completed)
            throw new InvalidOperationException("The encoding has already been completed.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(Base122Encoder));
    }
}
