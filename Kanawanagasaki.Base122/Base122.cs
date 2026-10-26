using System.Buffers;

namespace Kanawanagasaki.Base122;

/// <summary>
/// Base122 binary-to-text encoding: a UTF-8-safe encoding with only ~14% size
/// overhead over raw bytes (Base64 adds 33%). See
/// <see href="https://blog.kevinalbs.com/base122">the Base122 article</see>.
///
/// Six ASCII code units (NUL, LF, CR, double quote, ampersand, backslash) never
/// appear literally in the output; a 7-bit group that maps to one of them is
/// emitted as a single two-byte UTF-8 character together with the following
/// 7-bit group. The output of every encoding API is therefore valid UTF-8 text.
/// </summary>
public static class Base122
{
    private const int MaxStackallocChars = 512;
    private const int MaxStackallocBytes = 512;

    /// <summary>Inputs up to this size are encoded in place via <see cref="string.Create{TState}(int, TState, System.Buffers.SpanAction{char, TState})"/>; larger ones use a pooled buffer.</summary>
    private const int SingleAllocEncodeThreshold = 64 * 1024;

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

    private static FormatException InvalidUtf8Input(ReadOnlySpan<byte> encoded, int errorOffset)
    {
        byte b = errorOffset < encoded.Length ? encoded[errorOffset] : (byte)0;
        return new FormatException($"Invalid Base122 input: invalid byte 0x{b:X2} at index {errorOffset}.");
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

    private static FormatException InvalidInput(ReadOnlySpan<char> encoded, int errorOffset)
    {
        uint c = errorOffset < encoded.Length ? (uint)encoded[errorOffset] : 0;
        return new FormatException($"Invalid Base122 input: unexpected character U+{c:X4} at index {errorOffset}.");
    }
}
