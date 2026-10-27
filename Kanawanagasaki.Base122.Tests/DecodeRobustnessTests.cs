using Xunit;

namespace Kanawanagasaki.Base122.Tests;

/// <summary>
/// Robustness port of the reference suite's <c>test_decode_never_panics</c>:
/// random and corrupted inputs must never throw anything other than
/// <see cref="FormatException"/> (or return false from the Try API).
/// </summary>
public class DecodeRobustnessTests
{
    private static IEnumerable<char> PseudoRandomChars(ulong seed, int count)
    {
        for (int i = 0; i < count; i++)
            yield return (char)(TestHelpers.NextU64(ref seed) & 0x7FF);
    }

    [Fact]
    public void TryDecode_NeverThrows_OnArbitraryCharData()
    {
        Span<byte> buffer = new byte[4096];
        foreach (int len in new[] { 0, 1, 2, 3, 7, 8, 14, 15, 16, 31, 32, 63, 64, 255, 256, 511, 512, 1024 })
        {
            for (int trial = 0; trial < 100; trial++)
            {
                var chars = PseudoRandomChars(0xD1B5_4A32 + (ulong)len * 1000 + (ulong)trial, len).ToArray();
                // Any result is acceptable; a throw fails the test.
                Base122Decoder.TryDecode(chars, buffer, out _);
            }
        }
    }

    [Fact]
    public void TryDecode_NeverThrows_OnCorruptedValidEncodings()
    {
        Span<byte> buffer = new byte[8192];
        foreach (int len in new[] { 1, 8, 28, 57, 100, 1000 })
        {
            for (int trial = 0; trial < 200; trial++)
            {
                ulong seed = (ulong)len * 7919 + (ulong)trial;
                byte[] data = TestHelpers.RandomBytesArray(seed, len);
                var chars = Base122Encoder.Encode(data).ToArray();

                // Corrupt a character to a random value in the Base122 char space.
                int pos = (int)(TestHelpers.NextU64(ref seed) % (ulong)chars.Length);
                chars[pos] = (char)(TestHelpers.NextU64(ref seed) & 0x7FF);
                Base122Decoder.TryDecode(chars, buffer, out _);

                // Truncate the corrupted encoding.
                int cut = (int)(TestHelpers.NextU64(ref seed) % (ulong)(chars.Length + 1));
                Base122Decoder.TryDecode(chars.AsSpan(0, cut), buffer, out _);
            }
        }
    }

    [Fact]
    public void Decode_ThrowsOnlyFormatException_OnArbitraryCharData()
    {
        foreach (int len in new[] { 0, 1, 2, 3, 7, 8, 31, 63, 255, 511, 1024 })
        {
            for (int trial = 0; trial < 100; trial++)
            {
                var chars = PseudoRandomChars(0xABCD_1234 + (ulong)len * 31 + (ulong)trial, len).ToArray();
                try
                {
                    _ = Base122Decoder.Decode(chars);
                }
                catch (FormatException)
                {
                    // expected for invalid inputs
                }
            }
        }
    }

    [Fact]
    public void Encode_OutputIsAlwaysValidBase122CharSet()
    {
        foreach (int len in new[] { 0, 1, 2, 7, 8, 13, 14, 15, 16, 28, 29, 32, 56, 57, 100, 255, 1000, 4096 })
        {
            for (int trial = 0; trial < 25; trial++)
            {
                byte[] data = TestHelpers.RandomBytesArray((ulong)len * 131 + (ulong)trial, len);
                string encoded = Base122Encoder.Encode(data);

                foreach (char c in encoded)
                {
                    bool legal = c < 0x80 && TestHelpers.IllegalBytes.All(b => (char)b != c);
                    bool escape = c is >= '\u0080' and <= '\u07FF';
                    Assert.True(legal || escape, $"illegal character U+{c:X4} in output");
                }

                Assert.Equal(data, Base122Decoder.Decode(encoded));
            }
        }
    }
}
