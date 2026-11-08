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

Our upstream pull requests so far are listed under [Upstream pull requests](#upstream-pull-requests).

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

## Upstream pull requests

Our pull requests to [ergoplatform/sigma-rust](https://github.com/ergoplatform/sigma-rust), as of 2026-11-08. The live list is [on GitHub](https://github.com/ergoplatform/sigma-rust/pulls?q=is%3Apr+author%3Amwaddip).

### Merged (13)

| PR | Title | Into | Merged |
|---|---|---|---|
| [895](https://github.com/ergoplatform/sigma-rust/pull/895) | fix: Option DATA parser treats any nonzero tag as Some, per JVM getOption | `v0.30.0` | 2026-07-09 |
| [909](https://github.com/ergoplatform/sigma-rust/pull/909) | fix: read Header version as a signed Byte at the unparsedBytes gate (JVM parity) | `v0.30.0` | 2026-07-09 |
| [906](https://github.com/ergoplatform/sigma-rust/pull/906) | Reject self ContextExtension keys >= 0x80 (JVM context-construction parity) | `v0.30.0` | 2026-07-09 |
| [868](https://github.com/ergoplatform/sigma-rust/pull/868) | fix: reject non-pair tuples at eval to match sigma-state consensus | `v0.30.0` | 2026-07-09 |
| [877](https://github.com/ergoplatform/sigma-rust/pull/877) | fix(ergotree-ir): Global.powHit returns UnsignedBigInt, not Boolean | `v0.30.0` | 2026-07-09 |
| [897](https://github.com/ergoplatform/sigma-rust/pull/897) | fix: reject unsupported (arity != 2) tuple-typed values at eval (JVM checkType) | `v0.30.0` | 2026-07-09 |
| [891](https://github.com/ergoplatform/sigma-rust/pull/891) | fix: checked ERG summation in stateful transaction validation | `v0.30.0` | 2026-07-15 |
| [852](https://github.com/ergoplatform/sigma-rust/pull/852) | ergo-nipopow: tolerate skipped prefix entries in has_valid_connections | `v0.30.0` | 2026-08-03 |
| [866](https://github.com/ergoplatform/sigma-rust/pull/866) | fix(nipopow): pack_interlinks key encoding — use first-occurrence position | `v0.30.0` | 2026-08-03 |
| [915](https://github.com/ergoplatform/sigma-rust/pull/915) | style: restore fmt and clippy gates on v0.30.0 (ergo-nipopow) | `v0.30.0` | 2026-08-03 |
| [875](https://github.com/ergoplatform/sigma-rust/pull/875) | Set tree_version from the evaluated tree in reduce_to_crypto | `develop` | 2026-09-15 |
| [927](https://github.com/ergoplatform/sigma-rust/pull/927) | fix: wire-parse conformance with sigmastate 6.0.6 (folds #902, #911, #922, #926, #871) | `develop` | 2026-11-02 |
| [857](https://github.com/ergoplatform/sigma-rust/pull/857) | fix: BigInt256 modulo must use mathematical mod, not remainder | `develop` | 2026-11-02 |

### Open (43)

| PR | Title | Into | Opened |
|---|---|---|---|
| [847](https://github.com/ergoplatform/sigma-rust/pull/847) | Fix panic in gen_indexes when index modulo N equals zero | `develop` | 2026-03-31 |
| [848](https://github.com/ergoplatform/sigma-rust/pull/848) | ergo-lib: port ErgoTreePredef, EmissionRules, and genesis construction | `develop` | 2026-04-04 |
| [850](https://github.com/ergoplatform/sigma-rust/pull/850) | ergo-lib: add SoftForkVotesCollected and SoftForkStartingHeight Parameter variants | `develop` | 2026-04-06 |
| [851](https://github.com/ergoplatform/sigma-rust/pull/851) | ergo-nipopow: add prove_with_reader for db-backed proof construction | `develop` | 2026-04-08 |
| [855](https://github.com/ergoplatform/sigma-rust/pull/855) | fix: guard NiPoPow scorex_parse against allocation bomb | `v0.30.0` | 2026-04-09 |
| [858](https://github.com/ergoplatform/sigma-rust/pull/858) | perf: lazy constant resolution in ErgoTree evaluation *(draft)* | `develop` | 2026-04-12 |
| [859](https://github.com/ergoplatform/sigma-rust/pull/859) | fix: pre-JIT ErgoScript leniency for v0/v1 ErgoTree scripts | `develop` | 2026-04-13 |
| [860](https://github.com/ergoplatform/sigma-rust/pull/860) | fix: accept SOption(T) where T expected in check_post_eval_tpe | `develop` | 2026-04-16 |
| [865](https://github.com/ergoplatform/sigma-rust/pull/865) | fix(interpreter): wrap Resolver closures in Arc::new for upstream API change *(draft)* | `develop` | 2026-05-18 |
| [867](https://github.com/ergoplatform/sigma-rust/pull/867) | fix(interpreter): handle non-Spanned variants in wrap_spanned_with_src | `develop` | 2026-05-24 |
| [869](https://github.com/ergoplatform/sigma-rust/pull/869) | fix: coerce mismatched-numeric arithmetic operands to the wider type | `develop` | 2026-05-31 |
| [870](https://github.com/ergoplatform/sigma-rust/pull/870) | fix: negation of MIN_VALUE wraps to match sigma-state | `develop` | 2026-06-01 |
| [872](https://github.com/ergoplatform/sigma-rust/pull/872) | Fix v1 Autolykos zero pow_distance serialization to match sigma JVM | `develop` | 2026-06-03 |
| [873](https://github.com/ergoplatform/sigma-rust/pull/873) | Fix BigInt/UnsignedBigInt toBytes/toBits to emit minimal-length bytes | `develop` | 2026-06-03 |
| [874](https://github.com/ergoplatform/sigma-rust/pull/874) | Register GroupElement.expUnsigned in the method descriptor table | `develop` | 2026-06-03 |
| [876](https://github.com/ergoplatform/sigma-rust/pull/876) | JIT Costing: The Definitive PR *(draft)* | `develop` | 2026-06-03 |
| [878](https://github.com/ergoplatform/sigma-rust/pull/878) | fix(ergotree-interpreter): implement numeric cast to UnsignedBigInt in eval | `develop` | 2026-06-04 |
| [879](https://github.com/ergoplatform/sigma-rust/pull/879) | fix: tolerate absent/non-bytearray context var in DeserializeContext substitution | `develop` | 2026-06-04 |
| [880](https://github.com/ergoplatform/sigma-rust/pull/880) | fix(ergotree-interpreter): reduce atLeast with degenerate bound instead of erroring | `develop` | 2026-06-04 |
| [882](https://github.com/ergoplatform/sigma-rust/pull/882) | fix: deserialize FunDef (0xd7) — polymorphic val definitions | `develop` | 2026-06-06 |
| [883](https://github.com/ergoplatform/sigma-rust/pull/883) | fix: lambdas capture their defining environment | `develop` | 2026-06-06 |
| [884](https://github.com/ergoplatform/sigma-rust/pull/884) | fix: Box.getReg is method id 19; absent/out-of-range dynamic index is None | `develop` | 2026-06-06 |
| [885](https://github.com/ergoplatform/sigma-rust/pull/885) | fix: parse box value and token amount unbounded from the wire | `develop` | 2026-06-07 |
| [886](https://github.com/ergoplatform/sigma-rust/pull/886) | fix: AvlTree.insertOrUpdate returns None on a wrong-tree proof | `develop` | 2026-06-07 |
| [887](https://github.com/ergoplatform/sigma-rust/pull/887) | fix: Header.stateRoot is an AvlTree; v2 powOnetimePk is the group generator | `develop` | 2026-06-07 |
| [888](https://github.com/ergoplatform/sigma-rust/pull/888) | fix: EQ over cross-kind SigmaProps throws when the left is a conjecture | `develop` | 2026-06-07 |
| [889](https://github.com/ergoplatform/sigma-rust/pull/889) | fix: variable-length Context.headers; lastBlockUtxoRoot as its own context input | `develop` | 2026-06-07 |
| [890](https://github.com/ergoplatform/sigma-rust/pull/890) | fix: AvlTree ops map wrong-tree proofs and failed ops per the reference impl | `develop` | 2026-06-07 |
| [892](https://github.com/ergoplatform/sigma-rust/pull/892) | test: AvlTree degenerate-input eval regressions (pairs with ergo_avltree_rust#14) | `develop` | 2026-06-07 |
| [893](https://github.com/ergoplatform/sigma-rust/pull/893) | fix: reject standalone TreeLookup node at eval to match the reference interpreter | `develop` | 2026-06-07 |
| [894](https://github.com/ergoplatform/sigma-rust/pull/894) | fix: AvlTree digest is variable-length; updateDigest stores any-length Coll[Byte] | `develop` | 2026-06-07 |
| [898](https://github.com/ergoplatform/sigma-rust/pull/898) | fix: reject ErgoTree v>0 header without the size bit (JVM Rule-1012) | `develop` | 2026-06-10 |
| [899](https://github.com/ergoplatform/sigma-rust/pull/899) | fix: reject non-unary lambdas at closure creation (JVM FuncValue.eval) | `develop` | 2026-06-10 |
| [900](https://github.com/ergoplatform/sigma-rust/pull/900) | fix: Box/Header identity and Box.bytes use the retained wire bytes | `develop` | 2026-06-10 |
| [901](https://github.com/ergoplatform/sigma-rust/pull/901) | fix: deserialize LastBlockUtxoRootHash (0xa6) — the bare context-root op form | `develop` | 2026-06-10 |
| [903](https://github.com/ergoplatform/sigma-rust/pull/903) | Fix json+no_std build of ergotree-ir (missing alloc imports) | `develop` | 2026-06-12 |
| [905](https://github.com/ergoplatform/sigma-rust/pull/905) | Dispatch SBox accessor method-forms (PropertyCall 99:2..6) | `develop` | 2026-06-13 |
| [907](https://github.com/ergoplatform/sigma-rust/pull/907) | fix: reject two deserialize-time over-acceptances (v6 audit) | `develop` | 2026-06-14 |
| [908](https://github.com/ergoplatform/sigma-rust/pull/908) | fix: accept full u8 range for STypeVar name length (JVM TypeSerializer parity) | `develop` | 2026-06-15 |
| [910](https://github.com/ergoplatform/sigma-rust/pull/910) | fix: lossily decode STypeVar name to match the JVM TypeSerializer (UTF-8 leniency, byte-exact U+FFFD) | `develop` | 2026-06-16 |
| [923](https://github.com/ergoplatform/sigma-rust/pull/923) | fix: reject ContextExtension count >= 128 and variable id >= 0x80 at parse (sigmastate 6.0.6 parity) | `develop` | 2026-09-27 |
| [924](https://github.com/ergoplatform/sigma-rust/pull/924) | fix: storage-rent verdict is final; JVM Int fee/age arithmetic; StorageContractCost (ErgoInterpreter parity) | `develop` | 2026-09-27 |
| [925](https://github.com/ergoplatform/sigma-rust/pull/925) | fix: reject box creation height > Int.MaxValue at parse (getUIntExact parity) | `develop` | 2026-09-27 |

A further 10 were closed without merging: superseded, folded into a later pull request, or no longer needed.
