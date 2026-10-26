using System.Text;
using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>Cross-validation of the in-memory codec against the reference Rust implementation.</summary>
public class VectorTests
{
    public static IEnumerable<object[]> RustVectors() =>
        RustVectorData.Vectors.Select(v => new object[] { v.DataHex, v.EncodedHex });

    [Fact]
    public void Encode_HelloWorld_MatchesReference()
    {
        string encoded = Base122.Encode("hello world"u8);
        Assert.Equal("34192d46633c407737d78d4620", TestHelpers.ToHex(Encoding.UTF8.GetBytes(encoded)));
    }

    [Fact]
    public void Encode_Base122Word_MatchesReference()
    {
        string encoded = Base122.Encode("Base122"u8);
        Assert.Equal("21182e3629446432", TestHelpers.ToHex(Encoding.UTF8.GetBytes(encoded)));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void Encode_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);
        string expected = TestHelpers.EncodedVectorToText(encodedHex);

        Assert.Equal(expected, Base122.Encode(data));
        Assert.Equal(expected, Base122.Encode(data.AsSpan()));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void Decode_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] expected = TestHelpers.FromHex(dataHex);
        string encoded = TestHelpers.EncodedVectorToText(encodedHex);

        Assert.Equal(expected, Base122.Decode(encoded));
        Assert.Equal(expected, Base122.Decode(encoded.AsSpan()));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void TryDecode_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] expected = TestHelpers.FromHex(dataHex);
        string encoded = TestHelpers.EncodedVectorToText(encodedHex);

        Span<byte> buffer = new byte[Base122.GetMaxDecodedLength(encoded.Length) + 16];
        Assert.True(Base122.TryDecode(encoded, buffer, out int written));
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, buffer[..written].ToArray());
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void EncodedOutput_IsValidUtf8_AndNeverContainsIllegalCharacters(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);
        string encoded = Base122.Encode(data);

        // The encoded hex from Rust is valid UTF-8 by construction; comparing
        // byte forms proves the C# string is the same valid UTF-8 text.
        Assert.Equal(encodedHex, TestHelpers.ToHex(Encoding.UTF8.GetBytes(encoded)));

        foreach (byte illegal in TestHelpers.IllegalBytes)
            Assert.DoesNotContain((char)illegal, encoded);
    }
}

public class EncodeTests
{
    [Fact]
    public void Encode_Empty_ReturnsEmptyString()
    {
        Assert.Equal(string.Empty, Base122.Encode(Array.Empty<byte>()));
        Assert.Equal(string.Empty, Base122.Encode(Array.Empty<byte>().AsSpan()));
    }

    [Fact]
    public void Encode_NullArray_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Base122.Encode((byte[])null!));
    }

    [Fact]
    public void Encode_EverySingleByte_RoundTrips()
    {
        for (int b = 0; b <= 255; b++)
        {
            byte[] data = [(byte)b];
            Assert.Equal(data, Base122.Decode(Base122.Encode(data)));
        }
    }

    [Fact]
    public void Encode_AllIllegalBytes_MatchesReference()
    {
        byte[] data = TestHelpers.IllegalBytes;
        string encoded = Base122.Encode(data);
        Assert.Equal(TestHelpers.EncodedVectorToText("c2824152111938"), encoded);
    }

    [Fact]
    public void Encode_RepeatedZeros_HasExpectedLength()
    {
        // 100 zero bytes -> 57 escape pairs + 1 shortened escape = 58 characters.
        string encoded = Base122.Encode(new byte[100]);
        Assert.Equal(58, encoded.Length);
        Assert.Equal(58, Base122.GetEncodedLength(new byte[100]));
    }

    [Fact]
    public void Encode_OverloadsWithArrayAndSpan_Agree()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x1234, 1000);
        Assert.Equal(Base122.Encode(data), Base122.Encode(data.AsSpan()));
    }

    [Fact]
    public void TryEncode_MaxSizeBuffer_SucceedsWithExactCount()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x77, 300);
        int exact = Base122.GetEncodedLength(data);
        int max = Base122.GetMaxEncodedLength(data.Length);

        Span<char> buffer = new char[max];
        Assert.True(Base122.TryEncode(data, buffer, out int written));
        Assert.Equal(exact, written);
        Assert.Equal(Base122.Encode(data), new string(buffer[..written]));
    }

    [Fact]
    public void TryEncode_TooSmallBuffer_Fails()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x78, 100);
        Span<char> buffer = new char[Base122.GetEncodedLength(data) - 1];
        Assert.False(Base122.TryEncode(data, buffer, out int written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void TryEncode_TooSmallBuffer_MaxSizeSucceeds()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x79, 100);
        int max = Base122.GetMaxEncodedLength(data.Length);
        Assert.InRange(Base122.GetEncodedLength(data), 1, max);

        Span<char> buffer = new char[max];
        Assert.True(Base122.TryEncode(data, buffer, out int written));
        Assert.Equal(Base122.Encode(data).Length, written);
    }
}

public class DecodeTests
{
    [Fact]
    public void Decode_Empty_ReturnsEmptyArray()
    {
        Assert.Empty(Base122.Decode(string.Empty));
        Assert.Empty(Base122.Decode(ReadOnlySpan<char>.Empty));
    }

    [Fact]
    public void Decode_NullString_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Base122.Decode((string)null!));
    }

    public static IEnumerable<object[]> InvalidTexts()
    {
        // Chars in U+0100..U+017F map to invalid lead bytes 0xC4/0xC5.
        yield return new object[] { "\u0100" };
        yield return new object[] { "\u017F" };
        // U+0200..U+027F -> 0xC8/0xC9.
        yield return new object[] { "\u0200" };
        // U+0680..U+06FF -> 0xDA..0xDD (illegal index 6 and reserved).
        yield return new object[] { "\u0680" };
        yield return new object[] { "\u06FF" };
        // Anything >= U+0800 can never occur in Base122 text.
        yield return new object[] { "\u0800" };
        yield return new object[] { "\u20AC" };
        // Char with the mandatory 0x80 bit cleared inside the two-byte range.
        yield return new object[] { "\u0100" };
        yield return new object[] { "a\u0100b" };
    }

    [Theory]
    [MemberData(nameof(InvalidTexts))]
    public void Decode_InvalidCharacter_ThrowsFormat(string invalid)
    {
        Assert.Throws<FormatException>(() => Base122.Decode(invalid));
        Assert.Throws<FormatException>(() => Base122.Decode(invalid.AsSpan()));
    }

    [Theory]
    [MemberData(nameof(InvalidTexts))]
    public void TryDecode_InvalidCharacter_ReturnsFalse(string invalid)
    {
        Span<byte> buffer = new byte[64];
        Assert.False(Base122.TryDecode(invalid, buffer, out int written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void Decode_InvalidCharacter_ReportsPosition()
    {
        // "hello world" encoded, then a corrupted second character.
        string encoded = Base122.Encode("hello world"u8);
        var chars = encoded.ToArray();
        chars[1] = '\u0100';
        FormatException ex = Assert.Throws<FormatException>(() => Base122.Decode(new string(chars)));
        Assert.Contains("1", ex.Message);
    }

    [Fact]
    public void TryDecode_TooSmallBuffer_Fails()
    {
        string encoded = Base122.Encode(new byte[100]);
        Span<byte> buffer = new byte[Base122.Decode(encoded).Length - 1];
        Assert.False(Base122.TryDecode(encoded, buffer, out int written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void Decode_MaxSizeBuffer_WritesExactCount()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x42, 256);
        string encoded = Base122.Encode(data);
        int max = Base122.GetMaxDecodedLength(encoded.Length);
        Assert.InRange(data.Length, 1, max);

        Span<byte> buffer = new byte[max];
        Assert.True(Base122.TryDecode(encoded, buffer, out int written));
        Assert.Equal(data.Length, written);
        Assert.Equal(data, buffer[..written].ToArray());
    }

    [Fact]
    public void Decode_TruncatedPadding_DecodesShorterPrefix()
    {
        // A final shortened escape always decodes the full padded group; the
        // trailing partial byte is dropped, so decoding a valid prefix of an
        // encoding yields the original prefix bytes.
        byte[] data = [0x02];
        string encoded = Base122.Encode(data); // "\u0001\u0780"
        Assert.Equal(data, Base122.Decode(encoded));
    }
}

public class RoundTripTests
{
    [Fact]
    public void RoundTrip_ExhaustiveTwoByteInputs()
    {
        var buf = new byte[2];
        for (int a = 0; a <= 255; a++)
        {
            buf[0] = (byte)a;
            for (int b = 0; b <= 255; b++)
            {
                buf[1] = (byte)b;
                Assert.Equal(buf, Base122.Decode(Base122.Encode(buf)));
            }
        }
    }

    [Fact]
    public void RoundTrip_SampledThreeByteInputs()
    {
        var buf = new byte[3];
        for (int a = 0; a <= 255; a++)
        {
            buf[0] = (byte)a;
            for (int b = 0; b <= 255; b += 7)
            {
                buf[1] = (byte)b;
                for (int c = 0; c <= 255; c += 11)
                {
                    buf[2] = (byte)c;
                    Assert.Equal(buf, Base122.Decode(Base122.Encode(buf)));
                }
            }
        }
    }

    [Fact]
    public void RoundTrip_IllegalCombinationsExhaustively()
    {
        byte[] illegals = TestHelpers.IllegalBytes;
        byte[] edge = [0x00, 0x7F, 0x80, 0xFF];
        var values = illegals.Concat(edge).ToArray();

        foreach (byte a in values)
            foreach (byte b in values)
                foreach (byte c in values)
                {
                    byte[] data = [a, b, c];
                    Assert.Equal(data, Base122.Decode(Base122.Encode(data)));
                }
    }

    public static IEnumerable<object[]> LengthsAndPatterns()
    {
        foreach (int len in new[] { 0, 1, 2, 3, 6, 7, 8, 9, 13, 14, 15, 16, 27, 28, 29, 31, 32, 33, 63, 64, 65, 100, 255, 256, 511, 512, 1000 })
        {
            yield return new object[] { len, "random" };
            yield return new object[] { len, "zeros" };
            yield return new object[] { len, "illegal" };
            yield return new object[] { len, "alternating" };
            yield return new object[] { len, "ascii" };
        }
    }

    [Theory]
    [MemberData(nameof(LengthsAndPatterns))]
    public void RoundTrip_LengthSweep(int len, string pattern)
    {
        byte[] data = new byte[len];
        byte[] illegals = TestHelpers.IllegalBytes;
        for (int i = 0; i < len; i++)
        {
            switch (pattern)
            {
                case "zeros": data[i] = 0; break;
                case "illegal": data[i] = illegals[i % illegals.Length]; break;
                case "alternating": data[i] = (byte)(i % 2 == 0 ? illegals[i % illegals.Length] : i & 0xFF); break;
                case "ascii": data[i] = (byte)(32 + i % 95); break;
                default: data[i] = (byte)(i * 31 + 7); break;
            }
        }

        string encoded = Base122.Encode(data);
        byte[] decoded = Base122.Decode(encoded);

        Assert.Equal(data, decoded);
        Assert.InRange(encoded.Length, Base122.GetEncodedLength(data), Base122.GetMaxEncodedLength(data.Length));
        Assert.InRange(decoded.Length, 0, Base122.GetMaxDecodedLength(encoded.Length));
    }

    [Fact]
    public void RoundTrip_LargeRandomPayload()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xDEADBEEF, 1 << 20);
        string encoded = Base122.Encode(data);
        byte[] decoded = Base122.Decode(encoded);
        Assert.Equal(data, decoded);
    }

    [Fact]
    public void GetEncodedLength_MatchesActualLength_ForAllSmallPayloads()
    {
        for (int len = 1; len <= 64; len++)
        {
            for (int trial = 0; trial < 4; trial++)
            {
                byte[] data = TestHelpers.RandomBytesArray((ulong)len * 100 + (ulong)trial, len);
                if (trial == 3)
                    for (int i = 0; i < len; i++)
                        if (i % 2 == 0)
                            data[i] = TestHelpers.IllegalBytes[i % TestHelpers.IllegalBytes.Length];

                string encoded = Base122.Encode(data);
                Assert.Equal(encoded.Length, Base122.GetEncodedLength(data));
            }
        }
    }

    [Fact]
    public void GetMaxEncodedLength_BoundsActualLength()
    {
        for (int len = 1; len <= 512; len++)
        {
            byte[] data = TestHelpers.RandomBytesArray((ulong)len, len);
            int actual = Base122.Encode(data).Length;
            Assert.InRange(actual, 0, Base122.GetMaxEncodedLength(len));
        }

        // The bound is tight: 8 bytes -> 10 groups (64 bits) -> at most 10 characters.
        Assert.Equal(10, Base122.GetMaxEncodedLength(8));
        Assert.Equal(2, Base122.GetMaxEncodedLength(1));
        Assert.Equal(0, Base122.GetMaxEncodedLength(0));
    }

    [Fact]
    public void GetMaxDecodedLength_BoundsActualLength()
    {
        for (int len = 1; len <= 512; len++)
        {
            byte[] data = TestHelpers.RandomBytesArray((ulong)len + 999, len);
            string encoded = Base122.Encode(data);
            int actual = Base122.Decode(encoded).Length;
            Assert.InRange(actual, 0, Base122.GetMaxDecodedLength(encoded.Length));
        }

        Assert.Equal(0, Base122.GetMaxDecodedLength(0));
        Assert.Equal(14, Base122.GetMaxDecodedLength(8));
    }
}
