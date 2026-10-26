using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace Kanawanagasaki.Base122;

/// <summary>
/// Internal codec engine: shared lookup tables, SWAR (SIMD-within-a-register)
/// helpers and the exact-sized encode/decode kernels. Ported from the scalar
/// pipeline of the reference Rust implementation.
/// </summary>
internal static class Base122Core
{
    /// <summary>The six ASCII code units that must never appear literally in encoded output.</summary>
    public static readonly byte[] Illegals = [0, 10, 13, 34, 38, 92];

    /// <summary>0 = legal group, 1..6 = one-based illegal index. Indexed by 7-bit group value.</summary>
    public static readonly byte[] Class = BuildClass();

    /// <summary>
    /// Two-byte escape sequence (little-endian <see cref="ushort"/>) for an illegal group.
    /// Indexed by <c>((class - 1) &lt;&lt; 7) | nextGroup</c>.
    /// </summary>
    public static readonly ushort[] EscapePairs = BuildEscapePairs();

    /// <summary>Two-byte shortened escape sequence for a final zero-padded group (illegal index 7).</summary>
    public static readonly ushort[] Shortened = BuildShortened();

    /// <summary>Lead byte → <c>(illegalIndex &lt;&lt; 1) | firstBit</c>, or 0xFF when invalid.</summary>
    public static readonly byte[] LeadDecode = BuildLeadDecode();

    private static byte[] BuildClass()
    {
        var classes = new byte[128];
        for (int i = 0; i < Illegals.Length; i++)
            classes[Illegals[i]] = (byte)(i + 1);
        return classes;
    }

    private static ushort[] BuildEscapePairs()
    {
        var pairs = new ushort[6 * 128];
        for (int k = 0; k < 6; k++)
        {
            for (int next = 0; next < 128; next++)
            {
                byte lead = (byte)(0b1100_0010 | (k << 2) | (next >> 6));
                byte continuation = (byte)(0x80 | (next & 0x3F));
                pairs[(k << 7) | next] = (ushort)(lead | (continuation << 8));
            }
        }
        return pairs;
    }

    private static ushort[] BuildShortened()
    {
        var pairs = new ushort[128];
        for (int next = 0; next < 128; next++)
        {
            byte lead = (byte)(0b1100_0010 | (7 << 2) | (next >> 6));
            byte continuation = (byte)(0x80 | (next & 0x3F));
            pairs[next] = (ushort)(lead | (continuation << 8));
        }
        return pairs;
    }

    private static byte[] BuildLeadDecode()
    {
        var decode = new byte[256];
        Array.Fill(decode, (byte)0xFF);
        for (int k = 0; k < 6; k++)
        {
            for (int bit = 0; bit < 2; bit++)
                decode[0b1100_0010 | (k << 2) | bit] = (byte)((k << 1) | bit);
        }
        for (int bit = 0; bit < 2; bit++)
            decode[0b1100_0010 | (7 << 2) | bit] = (byte)((7 << 1) | bit);
        return decode;
    }

    // 0x7F in every byte lane.
    private const ulong LO7 = 0x7F7F7F7F7F7F7F7FUL;

    // 0x80 in every byte lane.
    private const ulong AsciiMask = 0x8080808080808080UL;

    private const ulong Z10 = 0x0A0A0A0A0A0A0A0AUL;
    private const ulong Z13 = 0x0D0D0D0D0D0D0D0DUL;
    private const ulong Z34 = 0x2222222222222222UL;
    private const ulong Z38 = 0x2626262626262626UL;
    private const ulong Z92 = 0x5C5C5C5C5C5C5C5CUL;

    /// <summary>
    /// Returns a per-byte 0x80 flag for every group byte equal to one of the six
    /// illegal values. All group bytes must be &lt; 0x80.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong IllegalHighMask(ulong groups)
    {
        // A lane matches a target t iff (x ^ t) == 0; zero-detection per lane via
        // z + 0x7F <= 0x7F (no cross-lane carry because z <= 0x7F).
        ulong t0 = ~(groups + LO7);
        ulong t10 = ~((groups ^ Z10) + LO7);
        ulong t13 = ~((groups ^ Z13) + LO7);
        ulong t34 = ~((groups ^ Z34) + LO7);
        ulong t38 = ~((groups ^ Z38) + LO7);
        ulong t92 = ~((groups ^ Z92) + LO7);
        return (t0 | t10 | t13 | t34 | t38 | t92) & AsciiMask;
    }

    /// <summary>
    /// Splits 56 bits (seven bytes, big-endian, in the low 56 bits) into eight
    /// consecutive 7-bit groups; group k is stored in byte k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Split56ToGroupsLE(ulong bits56)
    {
        // PDEP scatters the 56 bits into the low 7 bits of each byte lane
        // (reversed), so one deposit plus a byte swap yields all eight groups.
        if (Bmi2.X64.IsSupported)
            return BinaryPrimitives.ReverseEndianness(Bmi2.X64.ParallelBitDeposit(bits56, LO7));

        return ((bits56 >> 49) & 0x7F)
             | (((bits56 >> 42) & 0x7F) << 8)
             | (((bits56 >> 35) & 0x7F) << 16)
             | (((bits56 >> 28) & 0x7F) << 24)
             | (((bits56 >> 21) & 0x7F) << 32)
             | (((bits56 >> 14) & 0x7F) << 40)
             | (((bits56 >> 7) & 0x7F) << 48)
             | ((bits56 & 0x7F) << 56);
    }

    /// <summary>
    /// Extracts the eight 7-bit groups carried by the low seven bytes of a
    /// little-endian 64-bit load; group k is stored in byte k. The eighth byte
    /// is ignored (mirroring the reference codec's one-byte overreading loads).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong GroupsLE(ulong littleEndian)
    {
        // Reversing the endianness and dropping the low byte yields exactly the
        // low seven bytes packed big-endian into 56 bits.
        return Split56ToGroupsLE(BinaryPrimitives.ReverseEndianness(littleEndian) >> 8);
    }

    /// <summary>
    /// Packs eight 7-bit groups (group k in byte k) back into 56 bits, big-endian
    /// (the inverse of <see cref="GroupsLE"/>). Used by the decoding fast path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong GatherGroupsToBits56(ulong groupsLE)
    {
        // PEXT compacts the low 7 bits of each byte lane (reversed) back into
        // 56 consecutive bits.
        if (Bmi2.X64.IsSupported)
            return Bmi2.X64.ParallelBitExtract(BinaryPrimitives.ReverseEndianness(groupsLE), LO7);

        return ((groupsLE & 0xFF) << 49)
             | (((groupsLE >> 8) & 0xFF) << 42)
             | (((groupsLE >> 16) & 0xFF) << 35)
             | (((groupsLE >> 24) & 0xFF) << 28)
             | (((groupsLE >> 32) & 0xFF) << 21)
             | (((groupsLE >> 40) & 0xFF) << 14)
             | (((groupsLE >> 48) & 0xFF) << 7)
             | ((groupsLE >> 56) & 0xFF);
    }

    /// <summary>
    /// The encoded character for an illegal 7-bit group followed by the next
    /// 7-bit group: a two-byte UTF-8 character <c>U+0080 | (k &lt;&lt; 8) | next</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static char EscapeChar(int illegalValue, int next)
    {
        int k = Class[illegalValue] - 1;
        return (char)(0x80 | (k << 8) | next);
    }

    /// <summary>
    /// The encoded character for a final zero-padded group that would be illegal
    /// (the "shortened" escape, illegal index 7).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static char ShortenedChar(int group) => (char)(0x780 | group);

    /// <summary>
    /// Pulls the next 7-bit group from the bit stream. The final group (fewer
    /// than 7 bits left) is returned zero-padded; returns false when no bits remain.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Pull7(ReadOnlySpan<byte> data, ref int pos, ref ulong acc, ref int accBits, out int group)
    {
        while (accBits < 7 && pos < data.Length)
        {
            acc = (acc << 8) | data[pos++];
            accBits += 8;
        }

        if (accBits >= 7)
        {
            accBits -= 7;
            group = (int)((acc >> accBits) & 0x7F);
            acc &= (1UL << accBits) - 1;
            return true;
        }

        if (accBits > 0)
        {
            group = (int)((acc << (7 - accBits)) & 0x7F);
            acc = 0;
            accBits = 0;
            return true;
        }

        group = 0;
        return false;
    }

    /// <summary>
    /// Encodes <paramref name="data"/> into <paramref name="destination"/>, which
    /// must be at least <see cref="Base122.GetEncodedLength"/> characters long.
    /// Returns the number of characters written.
    /// </summary>
    public static int Encode(ReadOnlySpan<byte> data, Span<char> destination)
    {
        int len = data.Length;
        int i = 0;
        int pos = 0;
        bool hasPending = false;
        int pending = 0;

        // 8 groups at a time from 7-byte chunks, using safe 64-bit loads (the
        // eighth byte of each load is ignored). Escape-free chunks are stored
        // directly; chunks containing escapes go through the group machine.
        while (i + 8 <= len)
        {
            ulong groups = GroupsLE(BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(i, 8)));

            if (!hasPending && IllegalHighMask(groups) == 0)
            {
                StoreGroupsLE(destination, pos, groups, 8);
                pos += 8;
            }
            else
            {
                ProcessGroups(groups, 8, destination, ref pos, ref hasPending, ref pending);
            }

            i += 7;
        }

        // Remaining bytes (at most 7) go through a bit accumulator.
        ulong acc = 0;
        int accBits = 0;

        if (hasPending)
        {
            if (Pull7(data, ref i, ref acc, ref accBits, out int next))
                destination[pos++] = EscapeChar(pending, next);
            else
                destination[pos++] = ShortenedChar(pending);
            hasPending = false;
        }

        while (Pull7(data, ref i, ref acc, ref accBits, out int group))
        {
            if (Class[group] != 0)
            {
                if (Pull7(data, ref i, ref acc, ref accBits, out int next))
                    destination[pos++] = EscapeChar(group, next);
                else
                {
                    destination[pos++] = ShortenedChar(group);
                    break;
                }
            }
            else
            {
                destination[pos++] = (char)group;
            }
        }

        return pos;
    }

    /// <summary>
    /// Runs the per-group encoding machine over <paramref name="count"/> groups.
    /// An illegal group is remembered and paired with the following group into a
    /// single escape character.
    /// </summary>
    private static void ProcessGroups(ulong groups, int count, Span<char> destination, ref int pos, ref bool hasPending, ref int pending)
    {
        for (int k = 0; k < count; k++)
        {
            int group = (int)((groups >> (k << 3)) & 0xFF);

            if (hasPending)
            {
                destination[pos++] = EscapeChar(pending, group);
                hasPending = false;
            }
            else if (Class[group] != 0)
            {
                pending = group;
                hasPending = true;
            }
            else
            {
                destination[pos++] = (char)group;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreGroupsLE(Span<char> destination, int pos, ulong groups, int count)
    {
        for (int k = 0; k < count; k++)
            destination[pos + k] = (char)((groups >> (k << 3)) & 0xFF);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreGroupsLE(Span<byte> destination, int pos, ulong groups)
    {
        destination[pos] = (byte)groups;
        destination[pos + 1] = (byte)(groups >> 8);
        destination[pos + 2] = (byte)(groups >> 16);
        destination[pos + 3] = (byte)(groups >> 24);
        destination[pos + 4] = (byte)(groups >> 32);
        destination[pos + 5] = (byte)(groups >> 40);
        destination[pos + 6] = (byte)(groups >> 48);
        destination[pos + 7] = (byte)(groups >> 56);
    }

    /// <summary>
    /// Decodes <paramref name="encoded"/> into <paramref name="destination"/>,
    /// which must be at least <see cref="Base122.GetMaxDecodedLength"/>
    /// characters long. Returns false and reports the offending position when
    /// the input contains characters that cannot occur in Base122 text.
    /// </summary>
    public static bool Decode(ReadOnlySpan<char> encoded, Span<byte> destination, out int bytesWritten, out int errorOffset)
    {
        int pos = 0;
        ulong acc = 0;
        int accBits = 0;

        for (int i = 0; i < encoded.Length; i++)
        {
            uint c = encoded[i];

            if (c < 0x80)
            {
                // Plain character: 7 bits of data.
                acc = (acc << 7) | c;
                accBits += 7;
            }
            else if (c < 0x800 && (c & 0x80) != 0)
            {
                // Two-byte UTF-8 character: an escape carrying an illegal index
                // (kind 0..5, or 7 for the shortened ending) plus 7 data bits.
                uint kind = c >> 8;
                uint next = c & 0x7F;

                if (kind < 6)
                {
                    acc = (acc << 14) | ((uint)Illegals[(int)kind] << 7) | next;
                    accBits += 14;
                }
                else if (kind == 7)
                {
                    acc = (acc << 7) | next;
                    accBits += 7;
                }
                else
                {
                    bytesWritten = 0;
                    errorOffset = i;
                    return false;
                }
            }
            else
            {
                bytesWritten = 0;
                errorOffset = i;
                return false;
            }

            while (accBits >= 8)
            {
                accBits -= 8;
                destination[pos++] = (byte)(acc >> accBits);
            }
            acc &= (1UL << accBits) - 1;
        }

        // Fewer than 8 bits may remain; they are the zero padding of the final
        // group and are dropped, exactly reproducing the original byte count.
        bytesWritten = pos;
        errorOffset = 0;
        return true;
    }

    #region Byte-domain kernels (UTF-8 in and out; no chars or strings ever materialized)

    /// <summary>
    /// Encodes <paramref name="data"/> into the UTF-8 byte form of the Base122
    /// text. <paramref name="destination"/> must be at least
    /// <see cref="Base122.GetMaxEncodedToUtf8Length"/> bytes long. Returns the
    /// number of bytes written.
    /// </summary>
    public static int EncodeToUtf8(ReadOnlySpan<byte> data, Span<byte> destination)
    {
        int len = data.Length;
        int i = 0;
        int pos = 0;
        bool hasPending = false;
        int pending = 0;

        // 8 groups at a time from 7-byte chunks; escape-free chunks are stored
        // as 8 raw bytes directly.
        while (i + 8 <= len)
        {
            ulong groups = GroupsLE(BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(i, 8)));

            if (!hasPending && IllegalHighMask(groups) == 0)
            {
                destination[pos] = (byte)groups;
                destination[pos + 1] = (byte)(groups >> 8);
                destination[pos + 2] = (byte)(groups >> 16);
                destination[pos + 3] = (byte)(groups >> 24);
                destination[pos + 4] = (byte)(groups >> 32);
                destination[pos + 5] = (byte)(groups >> 40);
                destination[pos + 6] = (byte)(groups >> 48);
                destination[pos + 7] = (byte)(groups >> 56);
                pos += 8;
            }
            else
            {
                ProcessGroupsToBytes(groups, 8, destination, ref pos, ref hasPending, ref pending);
            }

            i += 7;
        }

        // Remaining bytes (at most 7) go through a bit accumulator.
        ulong acc = 0;
        int accBits = 0;

        if (hasPending)
        {
            if (Pull7(data, ref i, ref acc, ref accBits, out int next))
                WritePairBytes(destination, ref pos, pending, next);
            else
                WriteShortenedBytes(destination, ref pos, pending);
            hasPending = false;
        }

        while (Pull7(data, ref i, ref acc, ref accBits, out int group))
        {
            if (Class[group] != 0)
            {
                if (Pull7(data, ref i, ref acc, ref accBits, out int next))
                    WritePairBytes(destination, ref pos, group, next);
                else
                {
                    WriteShortenedBytes(destination, ref pos, group);
                    break;
                }
            }
            else
            {
                destination[pos++] = (byte)group;
            }
        }

        return pos;
    }

    /// <summary>
    /// Runs the per-group encoding machine over <paramref name="count"/> groups,
    /// emitting UTF-8 bytes: one for a legal group, two for an escape pair.
    /// </summary>
    private static void ProcessGroupsToBytes(ulong groups, int count, Span<byte> destination, ref int pos, ref bool hasPending, ref int pending)
    {
        for (int k = 0; k < count; k++)
        {
            int group = (int)((groups >> (k << 3)) & 0xFF);

            if (hasPending)
            {
                WritePairBytes(destination, ref pos, pending, group);
                hasPending = false;
            }
            else if (Class[group] != 0)
            {
                pending = group;
                hasPending = true;
            }
            else
            {
                destination[pos++] = (byte)group;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WritePairBytes(Span<byte> destination, ref int pos, int illegalValue, int next)
    {
        ushort pair = EscapePairs[((Class[illegalValue] - 1) << 7) | next];
        destination[pos++] = (byte)pair;
        destination[pos++] = (byte)(pair >> 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteShortenedBytes(Span<byte> destination, ref int pos, int group)
    {
        ushort pair = Shortened[group];
        destination[pos++] = (byte)pair;
        destination[pos++] = (byte)(pair >> 8);
    }

    /// <summary>
    /// Decodes the UTF-8 byte form of Base122 text into <paramref name="destination"/>.
    /// Returns false and reports the offending position when the input contains
    /// an invalid lead byte, an invalid continuation byte, or an escape sequence
    /// truncated by the end of the input.
    /// </summary>
    /// <remarks>
    /// <paramref name="destination"/> must be able to hold the decoded output:
    /// at most <see cref="Base122.GetMaxDecodedFromUtf8Length"/> bytes, since
    /// every input byte carries exactly 7 bits.
    /// </remarks>
    public static bool DecodeFromUtf8(ReadOnlySpan<byte> encoded, Span<byte> destination, out int bytesWritten, out int errorOffset)
    {
        int len = encoded.Length;
        int i = 0;
        int pos = 0;
        ulong acc = 0;
        int accBits = 0;

        while (i < len)
        {
            // Fast path: a window of eight consecutive clean (ASCII) bytes
            // decodes to seven output bytes in one shot.
            if (accBits == 0 && i + 8 <= len && pos + 7 <= destination.Length)
            {
                ulong chunk = BinaryPrimitives.ReadUInt64LittleEndian(encoded.Slice(i, 8));
                if ((chunk & AsciiMask) == 0)
                {
                    ulong bits56 = GatherGroupsToBits56(chunk);
                    destination[pos] = (byte)(bits56 >> 48);
                    destination[pos + 1] = (byte)(bits56 >> 40);
                    destination[pos + 2] = (byte)(bits56 >> 32);
                    destination[pos + 3] = (byte)(bits56 >> 24);
                    destination[pos + 4] = (byte)(bits56 >> 16);
                    destination[pos + 5] = (byte)(bits56 >> 8);
                    destination[pos + 6] = (byte)bits56;
                    pos += 7;
                    i += 8;
                    continue;
                }
            }

            int b = encoded[i];
            if (b < 0x80)
            {
                i++;
                acc = (acc << 7) | (uint)b;
                accBits += 7;
            }
            else
            {
                int code = LeadDecode[b];
                if (code == 0xFF)
                {
                    bytesWritten = 0;
                    errorOffset = i;
                    return false;
                }
                if (i + 1 >= len)
                {
                    bytesWritten = 0;
                    errorOffset = i;
                    return false;
                }
                int continuation = encoded[i + 1];
                if ((continuation & 0xC0) != 0x80)
                {
                    bytesWritten = 0;
                    errorOffset = i + 1;
                    return false;
                }
                i += 2;

                int next = ((code & 1) << 6) | (continuation & 0x3F);
                if ((code >> 1) < 6)
                {
                    acc = (acc << 14) | ((uint)Illegals[code >> 1] << 7) | (uint)next;
                    accBits += 14;
                }
                else
                {
                    acc = (acc << 7) | (uint)next;
                    accBits += 7;
                }
            }

            while (accBits >= 8)
            {
                accBits -= 8;
                destination[pos++] = (byte)(acc >> accBits);
            }
            acc &= (1UL << accBits) - 1;
        }

        // Fewer than 8 bits may remain; they are the zero padding of the final
        // group and are dropped, exactly reproducing the original byte count.
        bytesWritten = pos;
        errorOffset = 0;
        return true;
    }

    /// <summary>
    /// Computes the decoded byte count of a canonical UTF-8 encoding in O(1):
    /// every input byte carries exactly 7 bits, except that a (single, ending)
    /// shortened escape carries 7 bits per two bytes.
    /// </summary>
    /// <remarks>
    /// For non-canonical input this is still a safe upper bound for any input
    /// the decoder accepts, because every additional shortened escape only
    /// reduces the number of produced bits.
    /// </remarks>
    public static int GetDecodedFromUtf8Length(ReadOnlySpan<byte> encoded)
    {
        int len = encoded.Length;
        if (len == 0)
            return 0;

        // The last character is a two-byte escape iff the last byte has the
        // high bit set; it is the shortened escape iff its lead is DE or DF.
        bool shortened = len >= 2 && encoded[len - 1] >= 0x80 && (encoded[len - 2] == 0xDE || encoded[len - 2] == 0xDF);
        return (int)((7L * len - 7 * (shortened ? 1 : 0)) / 8);
    }

    #endregion
}
