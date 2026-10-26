use base122_fast::{decode_into, encode_into};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::RngExt;
use std::hint::black_box;

fn random_data(size: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut data = vec![0u8; size];
    rng.fill(&mut data[..]);
    data
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    let sizes = [16, 64, 256, 1024, 65536, 1_048_576];

    for &size in &sizes {
        let data = random_data(size);
        let mut out = Vec::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{}_bytes", size), |b| {
            b.iter(|| {
                let n = encode_into(black_box(&data), &mut out);
                black_box(n);
            });
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    let sizes = [16, 64, 256, 1024, 65536, 1_048_576];

    for &size in &sizes {
        let data = random_data(size);
        let encoded = base122_fast::encode(&data);
        let mut out = Vec::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{}_bytes", size), |b| {
            b.iter(|| {
                let n = decode_into(black_box(&encoded), &mut out).expect("decode failed");
                black_box(n);
            });
        });
    }
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    let sizes = [16, 64, 256, 1024, 65536, 1_048_576];

    for &size in &sizes {
        let data = random_data(size);
        let mut enc = Vec::new();
        let mut dec = Vec::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{}_bytes", size), |b| {
            b.iter(|| {
                let n = encode_into(black_box(&data), &mut enc);
                let s = unsafe { std::str::from_utf8_unchecked(&enc[..n]) };
                let m = decode_into(black_box(s), &mut dec).expect("roundtrip failed");
                black_box(m);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
