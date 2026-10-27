using System.Buffers;
using System.Buffers.Binary;

namespace Kanawanagasaki.Base122;

/// <summary>
/// Base122 binary-to-text encoding as a write-only <see cref="Stream"/>:
/// raw bytes written to this stream are encoded as Base122 text (UTF-8) into
/// the output stream supplied at construction.
/// <para>
/// Base122 is a UTF-8-safe encoding with only ~14% size overhead over raw
/// bytes (Base64 adds 33%). See
/// <see href="https://blog.kevinalbs.com/base122">the Base122 article</see>.
/// Six ASCII code units (NUL, LF, CR, double quote, ampersand, backslash) never
/// appear literally in the output; a 7-bit group that maps to one of them is
/// emitted as a single two-byte UTF-8 character together with the following
/// 7-bit group. The output of every encoding API is therefore valid UTF-8 text.
/// </para>
/// The class also hosts the static in-memory encode API
/// (<see cref="Encode(byte[])"/>, <see cref="EncodeToUtf8(ReadOnlySpan{byte})"/>,
/// including Stream, <see cref="IBufferWriter{Byte}"/> and async destinations).
/// </summary>
/// <remarks>
/// Because the class is a <see cref="Stream"/>, other streams can pipe into
/// it (e.g. <c>source.CopyTo(encoder)</c> or <c>await source.CopyToAsync(encoder)</c>).
/// <para>
/// The encoded bit stream may only be finalized once all input has been
/// written: the final (possibly zero-padded) group and any trailing escape are
/// emitted by <see cref="Complete"/> or <see cref="CompleteAsync"/> — which
/// <see cref="Dispose(bool)"/> and <see cref="DisposeAsync"/> call implicitly.
/// <see cref="Reset"/> revives a completed encoder for the next chunk.
/// </para>
/// Instances are not thread-safe.
/// </remarks>
public sealed class Base122Encoder : Stream
{
    private const int DefaultBufferSize = 8192;
    private const int MaxStackallocChars = 512;

    /// <summary>Inputs up to this size are encoded in place via <see cref="string.Create{TState}(int, TState, System.Buffers.SpanAction{char, TState})"/>; larger ones use a pooled buffer.</summary>
    private const int SingleAllocEncodeThreshold = 64 * 1024;

    private readonly Stream _output;
    private byte[] _buffer = null!;
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
    public Base122Encoder(Stream output) : this(output, null, leaveOpen: false) { }

    /// <summary>Creates an encoder that writes Base122 text (UTF-8) to <paramref name="output"/>.</summary>
    /// <param name="output">The stream receiving the encoded text.</param>
    /// <param name="leaveOpen">When true, the output stream outlives this encoder.</param>
    public Base122Encoder(Stream output, bool leaveOpen) : this(output, null, leaveOpen) { }

    /// <summary>Creates an encoder that writes Base122 text (UTF-8) to <paramref name="output"/>.</summary>
    /// <param name="output">The stream receiving the encoded text.</param>
    /// <param name="buffer">
    /// A caller-provided buffer used to stage encoded output before it is
    /// written to <paramref name="output"/> — pass e.g.
    /// <c>ArrayPool&lt;byte&gt;.Shared.Rent(8192)</c> and return it after
    /// disposal to avoid library-side allocations entirely. When null, an
    /// 8192-byte buffer is allocated on first use and reused for the
    /// lifetime of the encoder.
    /// </param>
    /// <param name="leaveOpen">When true, the output stream outlives this encoder.</param>
    public Base122Encoder(Stream output, byte[]? buffer, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(output);
        if (buffer is { Length: 0 })
            throw new ArgumentOutOfRangeException(nameof(buffer), "The buffer must not be empty.");
        _output = output;
        _buffer = buffer!;
        _leaveOpen = leaveOpen;
    }

    /// <summary>Allocates the staging buffer on first use (unless provided by the caller).</summary>
    private void EnsureBuffer() => _buffer ??= new byte[DefaultBufferSize];

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
        if (data.IsEmpty)
            return;
        EnsureBuffer();

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
        if (buffer.IsEmpty)
            return ValueTask.CompletedTask;
        EnsureBuffer();
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
        EnsureBuffer();
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
    /// Clears the encoder state so the same instance can encode another chunk
    /// into the same output stream — e.g. interleaving your own framing bytes
    /// (borders, length prefixes) between encoded chunks.
    /// </summary>
    /// <remarks>
    /// Any buffered output is flushed to the underlying stream first; the bit
    /// stream of a chunk ended by <see cref="Complete"/> is already final, so
    /// <see cref="Reset"/> only revives the encoder. Unfinalized bit state
    /// (a chunk not ended by <see cref="Complete"/>) is discarded — call
    /// <see cref="Complete"/> instead when the trailing data should be
    /// encoded. The underlying stream is neither closed nor disposed.
    /// </remarks>
    public void Reset()
    {
        ThrowIfDisposed();
        FlushBuffer();
        _acc = 0;
        _accBits = 0;
        _hasPendingIllegal = false;
        _pendingIllegal = 0;
        _buffered = 0;
        _completed = false;
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

    #region Static in-memory encode API

    /// <summary>
    /// Encodes binary data into a Base122 string.
    /// The output is valid UTF-8 and adds ~14% size overhead (Base64 adds 33%).
    /// </summary>
    /// <remarks>
    /// Inputs up to 64 KB are encoded in place via <see cref="string.Create{TState}(int, TState, System.Buffers.SpanAction{char, TState})"/>
    /// after computing the exact encoded length — a single allocation. Larger
    /// inputs skip the length pre-pass and encode into a pooled buffer instead.
    /// </remarks>
    public static string Encode(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length == 0)
            return string.Empty;

        if (data.Length <= SingleAllocEncodeThreshold)
        {
            int charCount = GetEncodedLength(data.AsSpan());
            return string.Create(charCount, data, static (chars, source) =>
            {
                int written = Base122Core.Encode(source, chars);
                System.Diagnostics.Debug.Assert(written == chars.Length, "Base122 length estimate must be exact.");
            });
        }

        return Encode((ReadOnlySpan<byte>)data);
    }

    /// <summary>
    /// Encodes binary data into a Base122 string.
    /// The output is valid UTF-8 and adds ~14% size overhead (Base64 adds 33%).
    /// </summary>
    /// <remarks>
    /// The data is encoded into a pooled buffer sized for the worst case and
    /// copied into the result string without a length pre-pass.
    /// </remarks>
    public static string Encode(ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty)
            return string.Empty;

        int max = GetMaxEncodedLength(data.Length);

        if (max <= MaxStackallocChars)
        {
            Span<char> buffer = stackalloc char[MaxStackallocChars];
            int written = Base122Core.Encode(data, buffer);
            return new string(buffer[..written]);
        }

        char[] rented = ArrayPool<char>.Shared.Rent(max);
        try
        {
            int written = Base122Core.Encode(data, rented);
            return new string(rented, 0, written);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Encodes binary data into a caller-provided buffer.
    /// <paramref name="destination"/> must be at least
    /// <see cref="GetMaxEncodedLength"/> characters long.
    /// </summary>
    /// <param name="data">The binary data to encode.</param>
    /// <param name="destination">The buffer receiving the encoded characters.</param>
    /// <param name="charsWritten">The exact number of encoded characters on success.</param>
    /// <returns>False when <paramref name="destination"/> is too small.</returns>
    public static bool TryEncode(ReadOnlySpan<byte> data, Span<char> destination, out int charsWritten)
    {
        if (destination.Length < GetMaxEncodedLength(data.Length))
        {
            charsWritten = 0;
            return false;
        }

        charsWritten = data.IsEmpty ? 0 : Base122Core.Encode(data, destination);
        return true;
    }

    /// <summary>
    /// Encodes binary data into the UTF-8 byte form of a Base122 text —
    /// the same bytes <see cref="Encode(byte[])"/> would produce as a string,
    /// but without materializing any string or character data.
    /// </summary>
    /// <remarks>
    /// Inputs up to 64 KB are encoded into an exactly sized array after
    /// computing the exact encoded length — a single allocation. Larger inputs
    /// encode into a pooled buffer and are copied to the result.
    /// </remarks>
    public static byte[] EncodeToUtf8(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        return EncodeToUtf8(data.AsSpan());
    }

    /// <summary>
    /// Encodes binary data into the UTF-8 byte form of a Base122 text —
    /// the same bytes <see cref="Encode(ReadOnlySpan{byte})"/> would produce as
    /// a string, but without materializing any string or character data.
    /// </summary>
    public static byte[] EncodeToUtf8(ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty)
            return [];

        if (data.Length <= SingleAllocEncodeThreshold)
        {
            // Exact length in one pass, then a single right-sized allocation.
            byte[] result = new byte[GetEncodedToUtf8Length(data)];
            int written = Base122Core.EncodeToUtf8(data, result);
            System.Diagnostics.Debug.Assert(written == result.Length, "Base122 length estimate must be exact.");
            return result;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(GetMaxEncodedToUtf8Length(data.Length));
        try
        {
            int written = Base122Core.EncodeToUtf8(data, rented);
            return rented.AsSpan(0, written).ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Encodes binary data into a caller-provided buffer holding the UTF-8 byte
    /// form of the Base122 text.
    /// <paramref name="destination"/> must be at least
    /// <see cref="GetMaxEncodedToUtf8Length"/> bytes long.
    /// </summary>
    /// <param name="data">The binary data to encode.</param>
    /// <param name="destination">The buffer receiving the encoded UTF-8 bytes.</param>
    /// <param name="bytesWritten">The exact number of encoded bytes on success.</param>
    /// <returns>False when <paramref name="destination"/> is too small.</returns>
    public static bool TryEncodeToUtf8(ReadOnlySpan<byte> data, Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < GetMaxEncodedToUtf8Length(data.Length))
        {
            bytesWritten = 0;
            return false;
        }

        bytesWritten = data.IsEmpty ? 0 : Base122Core.EncodeToUtf8(data, destination);
        return true;
    }

    /// <summary>
    /// Encodes binary data and writes the UTF-8 byte form of the Base122 text
    /// to <paramref name="destination"/>.
    /// </summary>
    /// <param name="data">The binary data to encode.</param>
    /// <param name="destination">The stream receiving the encoded UTF-8 bytes.</param>
    /// <returns>The number of encoded bytes written to the stream.</returns>
    public static int EncodeToUtf8(ReadOnlySpan<byte> data, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite)
            throw new NotSupportedException("The destination stream must be writable.");
        if (data.IsEmpty)
            return 0;

        byte[] rented = ArrayPool<byte>.Shared.Rent(GetMaxEncodedToUtf8Length(data.Length));
        try
        {
            int written = Base122Core.EncodeToUtf8(data, rented);
            destination.Write(rented, 0, written);
            return written;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Encodes binary data and writes the UTF-8 byte form of the Base122 text
    /// to <paramref name="bufferWriter"/>.
    /// </summary>
    /// <param name="data">The binary data to encode.</param>
    /// <param name="bufferWriter">The writer receiving the encoded UTF-8 bytes.</param>
    /// <returns>The number of encoded bytes written to the writer.</returns>
    public static int EncodeToUtf8(ReadOnlySpan<byte> data, IBufferWriter<byte> bufferWriter)
    {
        ArgumentNullException.ThrowIfNull(bufferWriter);
        if (data.IsEmpty)
            return 0;

        Span<byte> span = bufferWriter.GetSpan(GetMaxEncodedToUtf8Length(data.Length));
        int written = Base122Core.EncodeToUtf8(data, span);
        bufferWriter.Advance(written);
        return written;
    }

    /// <summary>
    /// Encodes binary data and asynchronously writes the UTF-8 byte form of
    /// the Base122 text to <paramref name="destination"/>, using the
    /// encoder's buffered async writes.
    /// </summary>
    /// <param name="data">The binary data to encode.</param>
    /// <param name="destination">The stream receiving the encoded UTF-8 bytes.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    public static async Task EncodeToUtf8Async(ReadOnlyMemory<byte> data, Stream destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite)
            throw new NotSupportedException("The destination stream must be writable.");

        await using var encoder = new Base122Encoder(destination, leaveOpen: true);
        await encoder.WriteAsync(data, cancellationToken).ConfigureAwait(false);
        await encoder.CompleteAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Returns the maximum number of bytes encoding
    /// <paramref name="byteCount"/> bytes to the UTF-8 byte form can ever produce.
    /// </summary>
    /// <remarks>
    /// Every 7-bit group costs at most one byte, except a single shortened
    /// escape at the end which costs two.
    /// </remarks>
    public static int GetMaxEncodedToUtf8Length(int byteCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(byteCount);
        if (byteCount == 0)
            return 0;
        return checked((int)((8L * byteCount + 6) / 7)) + 1;
    }

    /// <summary>
    /// Returns the exact number of bytes <see cref="EncodeToUtf8(ReadOnlySpan{byte})"/>
    /// produces for the given data.
    /// </summary>
    public static int GetEncodedToUtf8Length(ReadOnlySpan<byte> data)
    {
        int len = data.Length;
        if (len == 0)
            return 0;

        int bytes = 0;
        bool hasPending = false;
        int i = 0;
        ulong acc = 0;
        int accBits = 0;

        while (true)
        {
            while (accBits < 7 && i < len)
            {
                acc = (acc << 8) | data[i++];
                accBits += 8;
            }

            if (accBits >= 7)
            {
                accBits -= 7;
                int group = (int)((acc >> accBits) & 0x7F);
                acc &= (1UL << accBits) - 1;

                if (hasPending)
                {
                    // One escape pair (two bytes) covers the pending group and
                    // this one.
                    bytes += 2;
                    hasPending = false;
                }
                else if (Base122Core.Class[group] != 0)
                {
                    hasPending = true;
                    continue;
                }
                else
                {
                    bytes++;
                }
            }
            else if (accBits > 0)
            {
                // Final zero-padded group: a plain character (one byte), or an
                // escape pair or shortened escape (two bytes).
                int group = (int)((acc << (7 - accBits)) & 0x7F);
                if (hasPending || Base122Core.Class[group] != 0)
                    bytes += 2;
                else
                    bytes++;
                accBits = 0;
                acc = 0;
                hasPending = false;
                break;
            }
            else
            {
                break;
            }
        }

        // An illegal final group is emitted as a shortened escape (two bytes).
        if (hasPending)
            bytes += 2;

        return bytes;
    }

    /// <summary>
    /// Returns the exact number of characters <see cref="Encode(ReadOnlySpan{byte})"/>
    /// produces for the given data.
    /// </summary>
    public static int GetEncodedLength(ReadOnlySpan<byte> data)
    {
        int len = data.Length;
        if (len == 0)
            return 0;

        int chars = 0;
        bool hasPending = false;
        int i = 0;
        ulong acc = 0;
        int accBits = 0;

        while (true)
        {
            while (accBits < 7 && i < len)
            {
                acc = (acc << 8) | data[i++];
                accBits += 8;
            }

            if (accBits >= 7)
            {
                accBits -= 7;
                int group = (int)((acc >> accBits) & 0x7F);
                acc &= (1UL << accBits) - 1;

                if (hasPending)
                {
                    // One escape character covers the pending group and this one.
                    hasPending = false;
                }
                else if (Base122Core.Class[group] != 0)
                {
                    hasPending = true;
                    continue;
                }

                chars++;
            }
            else if (accBits > 0)
            {
                // Final zero-padded group: a plain character or a shortened
                // escape — one character either way.
                chars++;
                accBits = 0;
                acc = 0;
                hasPending = false;
                break;
            }
            else
            {
                break;
            }
        }

        // An illegal final group is emitted as a shortened escape.
        if (hasPending)
            chars++;

        return chars;
    }

    /// <summary>
    /// Returns the maximum number of characters encoding
    /// <paramref name="byteCount"/> bytes can ever produce.
    /// </summary>
    public static int GetMaxEncodedLength(int byteCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(byteCount);
        if (byteCount == 0)
            return 0;
        return checked((int)((8L * byteCount + 6) / 7));
    }

    #endregion
}
