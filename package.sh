#!/bin/bash
set -xeuo pipefail

rustup install nightly-2023-02-02
rustup component add rust-src --toolchain nightly-2023-02-02-aarch64-apple-darwin
cargo test --features c-headers -- generate_headers

TARGETS=(
  'aarch64-apple-ios'
  'x86_64-apple-darwin'
  'aarch64-apple-darwin'
  'x86_64-apple-ios'
  'aarch64-apple-ios-sim'
)
for target in "${TARGETS[@]}"
do
  rustup target add $target
  cargo build --release --target $target
done

TARGETS=(
  'aarch64-apple-ios-macabi'
  'x86_64-apple-ios-macabi'
)
for target in "${TARGETS[@]}"
do
  cargo +nightly-2023-02-02 build -Z build-std --release --target $target
done

lipo -create \
  target/x86_64-apple-darwin/release/libbfes.a \
  target/aarch64-apple-darwin/release/libbfes.a \
  -output libbfes_macos.a
lipo -create \
  target/x86_64-apple-ios/release/libbfes.a \
  target/aarch64-apple-ios-sim/release/libbfes.a \
  -output libbfes_iossimulator.a
lipo -create \
  target/aarch64-apple-ios-macabi/release/libbfes.a \
  target/x86_64-apple-ios-macabi/release/libbfes.a \
  -output libbfes_maccatalyst.a
rm -rf BFES.xcframework
xcodebuild -create-xcframework \
  -library ./libbfes_macos.a \
  -headers ./include/ \
  -library ./libbfes_iossimulator.a \
  -headers ./include/ \
  -library ./target/aarch64-apple-ios/release/libbfes.a \
  -headers ./include/ \
  -library ./libbfes_maccatalyst.a \
  -headers ./include/ \
  -output BFES.xcframework
