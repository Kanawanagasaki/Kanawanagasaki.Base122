using System.Text;

using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>
/// Cross-validation of the UTF-8 byte-form APIs (<see cref="Base122Encoder.EncodeToUtf8"/>,
/// <see cref="Base122Decoder.DecodeFromUtf8"/>) against the reference Rust implementation
/// and the char-based APIs.
/// </summary>
public class Utf8VectorTests
{
    public static IEnumerable<object[]> RustVectors() =>
        RustVectorData.Vectors.Select(v => new object[] { v.DataHex, v.EncodedHex });

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void EncodeToUtf8_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);
        byte[] expected = TestHelpers.FromHex(encodedHex);

        Assert.Equal(expected, Base122Encoder.EncodeToUtf8(data));
        Assert.Equal(expected, Base122Encoder.EncodeToUtf8(data.AsSpan()));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void DecodeFromUtf8_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] expected = TestHelpers.FromHex(dataHex);
        byte[] encoded = TestHelpers.FromHex(encodedHex);

        Assert.Equal(expected, Base122Decoder.DecodeFromUtf8(encoded));
        Assert.Equal(expected, Base122Decoder.DecodeFromUtf8(encoded.AsSpan()));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void TryEncodeToUtf8_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);
        byte[] expected = TestHelpers.FromHex(encodedHex);
        byte[] buffer = new byte[Base122Encoder.GetMaxEncodedToUtf8Length(data.Length)];

        Assert.True(Base122Encoder.TryEncodeToUtf8(data, buffer, out int written));
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, buffer[..written].ToArray());
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void TryDecodeFromUtf8_MatchesRustReference(string dataHex, string encodedHex)
    {
        byte[] expected = TestHelpers.FromHex(dataHex);
        byte[] encoded = TestHelpers.FromHex(encodedHex);
        byte[] buffer = new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length)];

        Assert.True(Base122Decoder.TryDecodeFromUtf8(encoded, buffer, out int written));
        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, buffer[..written].ToArray());
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void GetEncodedToUtf8Length_IsExact(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);

        Assert.Equal(encodedHex.Length / 2, Base122Encoder.GetEncodedToUtf8Length(data));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void GetDecodedFromUtf8Length_IsExactForCanonicalInput(string dataHex, string encodedHex)
    {
        byte[] encoded = TestHelpers.FromHex(encodedHex);

        Assert.Equal(dataHex.Length / 2, Base122Decoder.GetDecodedFromUtf8Length(encoded));
    }

    [Theory]
    [MemberData(nameof(RustVectors))]
    public void ByteApis_AgreeWithCharApis(string dataHex, string encodedHex)
    {
        byte[] data = TestHelpers.FromHex(dataHex);
        byte[] encoded = TestHelpers.FromHex(encodedHex);

        Assert.Equal(encoded, Encoding.UTF8.GetBytes(Base122Encoder.Encode(data)));
        Assert.Equal(Base122Decoder.Decode(Encoding.UTF8.GetString(encoded)), Base122Decoder.DecodeFromUtf8(encoded));
    }

    [Fact]
    public void EncodeToUtf8_HelloWorld_MatchesReference()
    {
        Assert.Equal("34192d46633c407737d78d4620", TestHelpers.ToHex(Base122Encoder.EncodeToUtf8("hello world"u8)));
    }
}

/// <summary>Round-trips through the byte-domain kernels, including worst-case escape patterns.</summary>
public class Utf8RoundTripTests
{
    [Fact]
    public void EverySingleByte_RoundTrips()
    {
        for (int b = 0; b <= 255; b++)
        {
            byte[] data = [(byte)b];
            byte[] encoded = Base122Encoder.EncodeToUtf8(data);

            Assert.Equal(Base122Encoder.GetEncodedToUtf8Length(data), encoded.Length);
            Assert.Equal(Encoding.UTF8.GetBytes(Base122Encoder.Encode(data)), encoded);
            Assert.Equal(data, Base122Decoder.DecodeFromUtf8(encoded));
            Assert.Equal(data.Length, Base122Decoder.GetDecodedFromUtf8Length(encoded));
        }
    }

    [Fact]
    public void EveryTwoBytes_RoundTrips()
    {
        // Exercises every 7-bit group value (0..127) in every role: plain,
        // first and second of an escape pair, final zero-padded group, and
        // pending-final (shortened escape).
        byte[] pair = new byte[2];
        Span<byte> buffer = new byte[Base122Encoder.GetMaxEncodedToUtf8Length(2)];

        for (int v = 0; v <= 0xFFFF; v++)
        {
            pair[0] = (byte)(v >> 8);
            pair[1] = (byte)v;

            byte[] encoded = Base122Encoder.EncodeToUtf8(pair);
            Assert.Equal(Encoding.UTF8.GetBytes(Base122Encoder.Encode(pair)), encoded);
            Assert.Equal(pair, Base122Decoder.DecodeFromUtf8(encoded));

            Assert.True(Base122Encoder.TryEncodeToUtf8(pair, buffer, out int written));
            Assert.Equal(encoded.Length, written);

            Assert.Equal(pair, Base122Decoder.DecodeFromUtf8(encoded));
            Assert.Equal(pair.Length, Base122Decoder.GetDecodedFromUtf8Length(encoded));
        }
    }

    [Fact]
    public void RandomData_OfManyLengths_RoundTrips()
    {
        foreach (ulong seed in new[] { 0x5EEDUL, 0xDEADBEEFUL })
        {
            for (int len = 0; len <= 512; len++)
            {
                byte[] data = TestHelpers.RandomBytesArray(seed + (ulong)len, len);
                byte[] encoded = Base122Encoder.EncodeToUtf8(data);

                Assert.Equal(Base122Encoder.GetEncodedToUtf8Length(data), encoded.Length);
                Assert.True(encoded.Length <= Base122Encoder.GetMaxEncodedToUtf8Length(len));
                Assert.Equal(data, Base122Decoder.DecodeFromUtf8(encoded));
                Assert.Equal(data.Length, Base122Decoder.GetDecodedFromUtf8Length(encoded));
            }
        }
    }

    [Fact]
    public void UniformBytes_AllZero_AllOnes_RoundTrip()
    {
        // All-zero data maximizes escape pairs; all-0xFF maximizes plain runs.
        foreach (byte fill in new byte[] { 0x00, 0xFF, 0x0A, 0x92 })
        {
            for (int len = 1; len <= 256; len += 7)
            {
                byte[] data = Enumerable.Repeat(fill, len).ToArray();
                byte[] encoded = Base122Encoder.EncodeToUtf8(data);

                Assert.Equal(Base122Encoder.GetEncodedToUtf8Length(data), encoded.Length);
                Assert.Equal(data, Base122Decoder.DecodeFromUtf8(encoded));
            }
        }
    }

    [Fact]
    public void EncodeToUtf8_AroundSingleAllocThreshold_MatchesCharApi()
    {
        foreach (int len in new[] { 64 * 1024 - 1, 64 * 1024, 64 * 1024 + 1 })
        {
            byte[] data = TestHelpers.RandomBytesArray(0x7AB1E5UL, len);

            Assert.Equal(Encoding.UTF8.GetBytes(Base122Encoder.Encode(data)), Base122Encoder.EncodeToUtf8(data));
            Assert.Equal(data, Base122Decoder.DecodeFromUtf8(Base122Encoder.EncodeToUtf8(data)));
        }
    }

    [Fact]
    public void EncodeToUtf8_AgreesWithStreamEncoder()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x57EA4UL, 50_000);

        using var output = new MemoryStream();
        using (var encoder = new Base122Encoder(output, leaveOpen: true))
            encoder.Write(data);

        Assert.Equal(Base122Encoder.EncodeToUtf8(data), output.ToArray());
    }
}

/// <summary>Error handling and non-canonical input for the byte-domain decoder.</summary>
public class Utf8ErrorTests
{
    [Theory]
    [InlineData("C480")]      // valid UTF-8 (U+0100) but not a Base122 escape lead
    [InlineData("80")]        // bare continuation byte
    [InlineData("E0A080")]    // 3-byte UTF-8 lead can never start a Base122 escape
    [InlineData("41C480")]    // invalid lead after one plain byte
    [InlineData("C241")]      // invalid continuation
    [InlineData("41C241")]    // invalid continuation after a plain byte
    public void TryDecodeFromUtf8_InvalidInput_ReturnsFalse(string hex)
    {
        byte[] encoded = TestHelpers.FromHex(hex);
        byte[] buffer = new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length)];

        Assert.False(Base122Decoder.TryDecodeFromUtf8(encoded, buffer, out int written));
        Assert.Equal(0, written);
    }

    [Theory]
    [InlineData("C480")]
    [InlineData("80")]
    [InlineData("E0A080")]
    [InlineData("41C480")]
    [InlineData("C241")]
    [InlineData("41C241")]
    public void DecodeFromUtf8_InvalidInput_ThrowsFormatException(string hex)
    {
        Assert.Throws<FormatException>(() => Base122Decoder.DecodeFromUtf8(TestHelpers.FromHex(hex)));
    }

    [Fact]
    public void DecodeFromUtf8_TruncatedEscapeAtEnd_ThrowsFormatException()
    {
        Assert.Throws<FormatException>(() => Base122Decoder.DecodeFromUtf8([0x41, 0xC2]));
    }

    [Fact]
    public void TryDecodeFromUtf8_TruncatedEscapeAtEnd_ReturnsFalse()
    {
        Assert.False(Base122Decoder.TryDecodeFromUtf8([0x41, 0xC2], new byte[8], out int written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void TryDecodeFromUtf8_DestinationTooSmall_ReturnsFalse()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x5EA1UL, 100);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);

        Assert.False(Base122Decoder.TryDecodeFromUtf8(encoded, new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length) - 1], out int written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void TryEncodeToUtf8_DestinationSmallerThanMax_ReturnsFalse()
    {
        byte[] data = TestHelpers.RandomBytesArray(0x5EA1UL, 100);
        int exact = Base122Encoder.GetEncodedToUtf8Length(data);
        int max = Base122Encoder.GetMaxEncodedToUtf8Length(data.Length);

        // The Try API requires the conservative maximum; an exact-size buffer
        // (smaller than max whenever an escape pair or shortened escape adds a
        // byte) is reported as too small rather than risking a partial encode.
        if (exact < max)
        {
            Assert.False(Base122Encoder.TryEncodeToUtf8(data, new byte[exact], out int written));
            Assert.Equal(0, written);
        }

        Assert.True(Base122Encoder.TryEncodeToUtf8(data, new byte[max], out int maxWritten));
        Assert.Equal(exact, maxWritten);
    }

    [Fact]
    public void DecodeFromUtf8_NonCanonicalMidStreamShortened_TrimsResult()
    {
        // DE 80 is a shortened escape (7 zero bits) mid-stream, followed by
        // 'A' (0x41) as a plain group: 14 bits -> one byte 0x01 plus 6 padding
        // bits. The O(1) length formula overestimates by one; the decoder trims.
        byte[] encoded = [0xDE, 0x80, 0x41];

        Assert.Equal(2, Base122Decoder.GetDecodedFromUtf8Length(encoded));
        Assert.Equal(new byte[] { 0x01 }, Base122Decoder.DecodeFromUtf8(encoded));
    }

    [Fact]
    public void TryDecodeFromUtf8_NonCanonicalMidStreamShortened_Succeeds()
    {
        byte[] encoded = [0xDE, 0x80, 0x41];
        byte[] buffer = new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length)];

        Assert.True(Base122Decoder.TryDecodeFromUtf8(encoded, buffer, out int written));
        Assert.Equal(1, written);
        Assert.Equal(new byte[] { 0x01 }, buffer[..written].ToArray());
    }

    [Fact]
    public void DecodeFromUtf8_Fuzz_NeverThrowsOtherExceptions_AndAgreesWithTry()
    {
        foreach (ulong seed in new[] { 0xF0_0DUL, 0xBAD_5EEDUL, 0x1234_5678UL })
        {
            for (int len = 0; len <= 96; len++)
            {
                byte[] encoded = TestHelpers.RandomBytesArray(seed + (ulong)len, len);
                byte[] buffer = new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length) + 8];

                bool tried = Base122Decoder.TryDecodeFromUtf8(encoded, buffer, out int written);

                byte[] decoded;
                try
                {
                    decoded = Base122Decoder.DecodeFromUtf8(encoded);
                }
                catch (FormatException)
                {
                    Assert.False(tried);
                    continue;
                }
                catch (Exception e)
                {
                    Assert.Fail($"DecodeFromUtf8 threw {e.GetType().Name} for input {TestHelpers.ToHex(encoded)}");
                    return;
                }

                Assert.True(tried);
                Assert.Equal(decoded, buffer[..written].ToArray());
                Assert.True(decoded.Length <= Base122Decoder.GetMaxDecodedFromUtf8Length(len));
            }
        }
    }
}

/// <summary>Boundary behavior of the UTF-8 length helpers.</summary>
public class Utf8LengthTests
{
    [Fact]
    public void ZeroLength_ReturnsZero()
    {
        Assert.Equal(0, Base122Encoder.GetMaxEncodedToUtf8Length(0));
        Assert.Equal(0, Base122Decoder.GetMaxDecodedFromUtf8Length(0));
        Assert.Equal(0, Base122Encoder.GetEncodedToUtf8Length(ReadOnlySpan<byte>.Empty));
        Assert.Equal(0, Base122Decoder.GetDecodedFromUtf8Length(ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void NegativeCounts_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Base122Encoder.GetMaxEncodedToUtf8Length(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Base122Decoder.GetMaxDecodedFromUtf8Length(-1));
    }

    [Fact]
    public void MaxBounds_HoldForAllExhaustivePairs()
    {
        byte[] pair = new byte[2];
        for (int v = 0; v <= 0xFFFF; v++)
        {
            pair[0] = (byte)(v >> 8);
            pair[1] = (byte)v;

            int encoded = Base122Encoder.EncodeToUtf8(pair).Length;
            Assert.True(encoded <= Base122Encoder.GetMaxEncodedToUtf8Length(2));
            Assert.True(encoded >= Base122Encoder.GetEncodedToUtf8Length(pair));
        }
    }

    [Fact]
    public void EncodedLength_NeverExceedsMax_ForManySizes()
    {
        for (int len = 1; len <= 300; len++)
        {
            byte[] data = TestHelpers.RandomBytesArray(0xB0_0000UL + (ulong)len, len);
            int encoded = Base122Encoder.EncodeToUtf8(data).Length;

            Assert.True(encoded <= Base122Encoder.GetMaxEncodedToUtf8Length(len));
            Assert.Equal(encoded, Base122Encoder.GetEncodedToUtf8Length(data));
        }
    }
}

/// <summary>Empty-input and null-handling of the byte-form APIs.</summary>
public class Utf8EmptyInputTests
{
    [Fact]
    public void EncodeToUtf8_Empty_ReturnsEmpty()
    {
        Assert.Empty(Base122Encoder.EncodeToUtf8(Array.Empty<byte>()));
        Assert.Empty(Base122Encoder.EncodeToUtf8(ReadOnlySpan<byte>.Empty));
        Assert.Empty(Base122Encoder.EncodeToUtf8(new byte[0]));
    }

    [Fact]
    public void DecodeFromUtf8_Empty_ReturnsEmpty()
    {
        Assert.Empty(Base122Decoder.DecodeFromUtf8(Array.Empty<byte>()));
        Assert.Empty(Base122Decoder.DecodeFromUtf8(ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void Try_EmptyData_SucceedsWithZeroWritten()
    {
        Assert.True(Base122Encoder.TryEncodeToUtf8(ReadOnlySpan<byte>.Empty, Span<byte>.Empty, out int encoded));
        Assert.Equal(0, encoded);

        Assert.True(Base122Decoder.TryDecodeFromUtf8(ReadOnlySpan<byte>.Empty, Span<byte>.Empty, out int decoded));
        Assert.Equal(0, decoded);
    }

    [Fact]
    public void NullArray_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Base122Encoder.EncodeToUtf8((byte[])null!));
        Assert.Throws<ArgumentNullException>(() => Base122Decoder.DecodeFromUtf8((byte[])null!));
    }
}

/// <summary>Allocation behavior of the UTF-8 byte-form APIs.</summary>
public class Utf8AllocationTests
{
    private static long AllocatedBytes(Action action)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        action();

        long before = GC.GetAllocatedBytesForCurrentThread();
        action();
        long after = GC.GetAllocatedBytesForCurrentThread();
        return after - before;
    }

    [Fact]
    public void EncodeToUtf8_Small_AllocatesOnlyTheResult()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xA11CE, 10_000);
        byte[] expected = Base122Encoder.EncodeToUtf8(data);

        long allocated = AllocatedBytes(() => Base122Encoder.EncodeToUtf8(data));
        // Exactly one byte[] allocation (payload + object header) plus slack.
        Assert.InRange(allocated, expected.Length, expected.Length + 64);
    }

    [Fact]
    public void DecodeFromUtf8_Canonical_AllocatesOnlyTheResult()
    {
        byte[] encoded = Base122Encoder.EncodeToUtf8(TestHelpers.RandomBytesArray(0xB0B, 10_000));
        byte[] expected = Base122Decoder.DecodeFromUtf8(encoded);

        long allocated = AllocatedBytes(() => Base122Decoder.DecodeFromUtf8(encoded));
        Assert.InRange(allocated, expected.Length, expected.Length + 64);
    }

    [Fact]
    public void TryEncodeAndTryDecodeToUtf8_AllocateNothing()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xD1CE, 10_000);
        byte[] encoded = Base122Encoder.EncodeToUtf8(data);
        byte[] encodedBuffer = new byte[Base122Encoder.GetMaxEncodedToUtf8Length(data.Length)];
        byte[] decodedBuffer = new byte[Base122Decoder.GetMaxDecodedFromUtf8Length(encoded.Length)];

        long allocated = AllocatedBytes(() =>
        {
            Base122Encoder.TryEncodeToUtf8(data, encodedBuffer, out int bytesWritten);
            Base122Decoder.TryDecodeFromUtf8(encoded, decodedBuffer, out _);
        });

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void EncodeToUtf8_Large_AllocatesOnlyTheResult()
    {
        byte[] data = TestHelpers.RandomBytesArray(0xB0B, 100_000);
        byte[] expected = Base122Encoder.EncodeToUtf8(data);

        long allocated = AllocatedBytes(() => Base122Encoder.EncodeToUtf8(data));
        // Exact-length first: only the result array (~114 KB).
        Assert.InRange(allocated, expected.Length, expected.Length + 64);
    }
}
