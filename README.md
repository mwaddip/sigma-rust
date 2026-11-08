Rust implementation of [ErgoScript](https://github.com/ScorexFoundation/sigmastate-interpreter) cryptocurrency scripting language.

## About this fork

This is a fork of [ergoplatform/sigma-rust](https://github.com/ergoplatform/sigma-rust), not a competing
project. Think of it as a bleeding-edge build of sigma-rust, kept close to upstream `develop`, with three aims:

1. **Take pressure off upstream review.** We have a backlog of open pull requests upstream. Carrying them here
   lets work that depends on them continue meanwhile, so upstream reviewers can clear the queue at their own pace.
2. **Converge towards a validation-ready sigma-rust upstream.** The goal is a library that matches the reference
   implementation, [sigmastate](https://github.com/ScorexFoundation/sigmastate-interpreter), closely enough to
   validate blocks and transactions. That work happens here first and is meant to flow back upstream.
3. **Send cleaner upstream PRs later.** Once the backlog has shrunk, changes that have been running here can go
   upstream as consolidated, battle-tested pull requests rather than as a series of incremental fixes.

Releases (crates.io, npm, PyPI) come from upstream.

See [Architecture](docs/architecture.md) for high-level overview.

## Crates

[ergo-lib](ergo-lib)

Overarching crate exposing wallet-related features: chain types (transactions, boxes, etc.), JSON serialization, box selection for tx inputs, tx builder and signing. Exports other crates API, probably the only crate you'd need to import.

[ergotree-interpreter](ergotree-interpreter)

ErgoTree interpreter

[ergotree-ir](ergotree-ir)

ErgoTree IR and serialization.

[ergoscript-compiler](ergoscript-compiler)

ErgoScript compiler.

[sigma-ser](sigma-ser)

Ergo binary serialization primitives.

Bindings:

- [ergo-lib-wasm(Wasm)](bindings/ergo-lib-wasm)
- [ergo-lib-wasm-browser(JS/TS)](bindings/ergo-lib-wasm)
- [ergo-lib-wasm-nodejs(JS/TS)](bindings/ergo-lib-wasm)
- [ergo-lib-ios(Swift)](bindings/ergo-lib-ios)
- [ergo-lib-jni(Java)](bindings/ergo-lib-jni)
- [ergo-lib-c (C)](bindings/ergo-lib-c)
- [ergo-lib-go (Go)](https://github.com/sigmaspace-io/ergo-lib-go)
- [sigma_rb(Ruby)](https://github.com/thedlop/sigma_rb)
- [ergo-lib-python](bindings/ergo-lib-python)

## Changelog

See [CHANGELOG.md](ergo-lib/CHANGELOG.md).

## Usage Examples

To get better understanding on how to use it in your project check out how its being used in the following projects:

Rust:

- [Oracle Core](https://github.com/ergoplatform/oracle-core);
- [Ergo Headless dApp Framework](https://github.com/Emurgo/ergo-headless-dapp-framework);
- [Ergo Node Interface Library](https://github.com/Emurgo/ergo-node-interface);
- [Spectrum Off-Chain Services for Ergo](https://github.com/spectrum-finance/spectrum-offchain-ergo);
- [AgeUSD Stablecoin Protocol](https://github.com/Emurgo/age-usd);
- [ErgoNames SDKs](https://github.com/ergonames/sdk/tree/master/rust)

TS/JS:

- [Ergo SDK](https://github.com/ergolabs/ergo-sdk-js) (Wasm bindings);
- [Yoroi wallet](https://github.com/Emurgo/yoroi-frontend) (Wasm bindings);
- [Ergo Desktop Wallet](https://github.com/ErgoWallet/ergowallet-desktop) (Wasm bindings);

Examples:

- [Create transaction demo](bindings/ergo-lib-wasm/examples/create-transaction-demo) (TS)
- [Address generation demo](bindings/ergo-lib-wasm/examples/address-generation-demo) (TS)

Also take a look at tests where various usage scenarios were implemented.

## Contributing

See [Contributing](CONTRIBUTING.md) guide.

Feel free to join the [Ergo Discord](https://discord.gg/kj7s7nb) and ask questions on `#sigma-rust` channel.
