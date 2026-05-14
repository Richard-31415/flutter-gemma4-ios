# Gemma 4 iOS Demo — Flutter

A minimal Flutter app that runs **Gemma 4 E2B / E4B** entirely on an iPhone via [flutter_gemma](https://pub.dev/packages/flutter_gemma). Streams responses, parses Gemma 4's thinking-mode tokens into a separate panel, and works fully offline after a one-time model download.

Built and verified on **iPhone 16 / iOS 18.7.7** with a **free Apple Developer account**, using [GetMoreRam](https://github.com/hugeBlack/GetMoreRam) to unlock the memory entitlement that Gemma 4 needs.

---

## Why this exists

Getting Gemma 4 to run on iOS via flutter_gemma 0.15.0 is surprisingly fiddly even when the docs look straightforward. This repo documents the three gotchas that cost me most of a day and shows the minimum working code:

1. **Memory entitlement on free Apple Dev accounts** — Gemma 4 fails to load on GPU with `std::bad_alloc` unless the app has `com.apple.developer.kernel.increased-memory-limit`. Apple's Free tier does NOT grant this entitlement. → [GetMoreRam](https://github.com/hugeBlack/GetMoreRam) patches it onto your App ID at the developer-portal level, and the patch persists across `flutter run` rebuilds.
2. **`fileType: ModelFileType.litertlm`** must be passed explicitly to `installModel()` — flutter_gemma defaults to `ModelFileType.task`, which silently routes through the wrong chat-template path on iOS and crashes mid-prefill with a NULL `memset`.
3. **Thinking-mode tokens are not auto-parsed** — Gemma 4 emits `<|channel>thought\n...<channel|>` blocks, but flutter_gemma 0.15.0 doesn't auto-route them into `ThinkingResponse` events. You have to wrap the stream with `ModelThinkingFilter.filterThinkingStream()` manually.

All three are addressed in [lib/main.dart](lib/main.dart) with inline comments.

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS on Apple Silicon | Intel Macs are unsupported by flutter_gemma's iOS dev path |
| Xcode 16+ | iOS Platform package must be installed (Xcode → Settings → Platforms) |
| Flutter SDK 3.40+ | `brew install --cask flutter` (via native Homebrew at `/opt/homebrew`) |
| CocoaPods | `brew install --formula cocoapods` (system Ruby is too old) |
| Physical iPhone with iOS 16+ | Simulator has a 256 MB Metal allocation cap → Gemma 4 won't fit |
| Developer Mode enabled on iPhone | Settings → Privacy & Security → Developer Mode |
| Apple Developer account (free is fine) | Used for code signing |
| HuggingFace account | Token only needed for gated repos; the `litert-community` Gemma 4 repos are currently public |

---

## Setup

### 1. Install Flutter dependencies

```bash
flutter pub get
cd ios && LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 pod install && cd ..
```

The `LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` workaround is for a CocoaPods + Ruby 4 encoding bug. You can also export these in `~/.zshrc`.

### 2. Create your HuggingFace token file (optional)

```bash
cp config.json.example config.json
# Edit config.json and paste your hf_... token from
# https://huggingface.co/settings/tokens
```

Required scope: "Read access to contents of all public gated repos you can access".

`config.json` is gitignored.

### 3. Open the iOS project and set your signing team

```bash
open ios/Runner.xcworkspace
```

In Xcode:
- Click the **Runner** project icon → **Runner** target → **Signing & Capabilities**
- **Team**: select your personal Apple ID team
- **Bundle Identifier**: change `com.example.gemma4iosdemo` to something globally unique (Apple won't let two people sign the same bundle ID)

### 4. Unlock the memory entitlement (free Apple Dev only)

Skip this step if you have a paid Apple Developer Program membership and have added the "Increased Memory Limit" capability properly.

Otherwise: install [GetMoreRam](https://github.com/hugeBlack/GetMoreRam) via SideStore or AltStore, then in Get More Ram:
1. Sign in with the same Apple ID you used in Xcode
2. App IDs → Refresh → tap your bundle ID
3. Tap "Add Increased Memory Limit"

The entitlement is now attached to your App ID at Apple's portal. Every subsequent `flutter run` will pull a provisioning profile that includes it. You only do this once per bundle ID.

### 5. Build and run

```bash
flutter devices  # find your iPhone's UDID
flutter run -d <udid> --release --dart-define-from-file=config.json
```

Release mode is required for any meaningful inference performance — debug mode is unusably slow.

---

## Usage

1. App opens. Pick a model from the dropdown (default: Gemma 4 E2B, 2.4 GB).
2. Tap **Download + load**. First run downloads from HuggingFace (~5 min on home wifi). Subsequent runs are instant — the model file is cached in the app's Documents directory and persists across launches.
3. When status shows "Model ready.", tap **Run inference**.
4. The model's reasoning streams into the **Thinking** panel; the final answer streams into the **Answer** panel.

GPU prefill is ~0.3 s, decode is ~25–55 tok/s on iPhone 16 — basically real-time.

---

## What's in here

| File | Purpose |
|---|---|
| [`lib/main.dart`](lib/main.dart) | Whole app (~300 lines) — UI, model install, streaming inference with thinking filter |
| [`pubspec.yaml`](pubspec.yaml) | Single dep: `flutter_gemma: ^0.15.0` |
| [`ios/Podfile`](ios/Podfile) | `platform :ios, '16.0'` + `use_frameworks! :linkage => :static` |
| [`ios/Runner/Info.plist`](ios/Runner/Info.plist) | `UIFileSharingEnabled`, `NSLocalNetworkUsageDescription`, `CADisableMinimumFrameDurationOnPhone` |
| [`ios/Runner/Runner.entitlements`](ios/Runner/Runner.entitlements) | Memory entitlements — present but **not linked** in the Xcode project. Free Apple Dev accounts can't link it (Apple's portal will reject the provisioning request); use Get More Ram instead. Paid accounts can link it via Xcode → Runner target → Build Settings → "Code Signing Entitlements" |
| [`config.json.example`](config.json.example) | Template for your HF token |

No native code edits, no custom build phases.

---

## Tested configurations

| Device | iOS | Apple Dev | Backend | Status |
|---|---|---|---|---|
| iPhone 16 (8 GB RAM) | 18.7.7 | Free + Get More Ram | GPU (Metal) | ✅ E2B + E4B both work |

If you test on other devices, PRs to update this table are welcome.

---

## Credits

- [flutter_gemma](https://github.com/DenisovAV/flutter_gemma) by Mobile People — the plugin that makes this possible
- [GetMoreRam](https://github.com/hugeBlack/GetMoreRam) by hugeBlack / Stossy11 — the entitlement workaround for free-tier Apple Dev
- [LiteRT Community](https://huggingface.co/litert-community) — the `.litertlm` model packaging

## License

MIT
