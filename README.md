# Gemma 4 iOS Demo — Flutter

A minimal Flutter app that runs **Gemma 4 E2B / E4B** entirely on an iPhone via [flutter_gemma](https://pub.dev/packages/flutter_gemma). Streams responses, parses Gemma 4's thinking-mode tokens into a separate panel, accepts **native voice input** (audio → reasoning → answer in a single Gemma 4 model pass — no separate ASR), shows **live benchmarking** (TTFT, tok/s, RSS memory), and works fully offline after a one-time model download.

Built and verified on **iPhone 16 / iOS 18.7.7** with a **free Apple Developer account**.

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

### 4. Build and run

```bash
flutter devices  # find your iPhone's UDID
flutter run -d <udid> --release --dart-define-from-file=config.json
```

Release mode is required for any meaningful inference performance — debug mode is unusably slow.

---

## Usage

1. App opens. Pick a model from the dropdown (default: Gemma 4 E2B, 2.4 GB), then choose **maxTokens** (KV-cache budget; 1024 is a safe default for E4B on iPhone 16) and **MTP** (multi-token prediction / speculative decoding — `Default (model)` honors the `.litertlm` file's manifest, which is on for litert-community Gemma 4).
2. Tap **Download + load**. First run downloads from HuggingFace (~5 min on home wifi). Subsequent runs are instant — the model file is cached in the app's Documents directory and persists across launches. Status will show `Model ready. RSS idle: XXX MB` — that's the loaded-model memory footprint baseline. The three setting dropdowns now lock; engine settings are baked in at load time and the native model can't be unloaded in-process. **To change settings, force-quit the app and relaunch.**
3. Type a prompt and tap **Run (text)**, OR tap **Record + run audio** to capture mic input and send it directly to Gemma 4's audio encoder (Path A — no separate ASR).
4. The model's reasoning streams into the **Thinking** panel; the final answer streams into the **Answer** panel.
5. The **Stats** card shows live TTFT, total time, tok/s, token counts, and RSS idle vs peak. RSS is read from iOS's `task_vm_info.phys_footprint` via a small Swift platform channel ([`ios/Runner/AppDelegate.swift`](ios/Runner/AppDelegate.swift)) — this is the same number Xcode's Debug Navigator shows and what iOS's per-process memory cap is enforced against. (Dart's built-in `ProcessInfo.currentRss` returns `resident_size`, which on iOS excludes Metal/IOKit allocations and badly under-counts GPU-resident model weights.)
6. Tap **Copy all** to bundle config (model, maxTokens, MTP) + prompt + system + stats + thinking + answer into the clipboard — useful for sharing benchmark runs or comparing across configurations.

Tap-outside-textfield dismisses the keyboard; everything scrolls so the keyboard never covers content.

---

## What's in here

| File | Purpose |
|---|---|
| [`lib/main.dart`](lib/main.dart) | Whole app — UI (model + maxTokens + MTP dropdowns, prompt fields, stats card, output panes), model install, streaming inference with thinking filter, native audio input, in-app benchmarking (TTFT/tok-per-sec/RSS) |
| [`ios/Runner/AppDelegate.swift`](ios/Runner/AppDelegate.swift) | Registers a tiny `gemma4_demo/memory_stats` method channel that returns `task_vm_info.phys_footprint` — the number iOS's memory cap is enforced against, including Metal/IOKit memory where Gemma's GPU weights live |
| [`pubspec.yaml`](pubspec.yaml) | Deps: `flutter_gemma`, `record` (mic), `path_provider` (temp WAV file) |
| [`ios/Podfile`](ios/Podfile) | `platform :ios, '16.0'` + `use_frameworks! :linkage => :static` |
| [`ios/Runner/Info.plist`](ios/Runner/Info.plist) | `UIFileSharingEnabled`, `NSLocalNetworkUsageDescription`, `NSMicrophoneUsageDescription`, `CADisableMinimumFrameDurationOnPhone` |
| [`ios/Runner/Runner.entitlements`](ios/Runner/Runner.entitlements) | Memory entitlements — present but **not linked** in the Xcode project. Paid Apple Developer accounts can link it via Xcode → Runner target → Build Settings → "Code Signing Entitlements". |
| [`config.json.example`](config.json.example) | Template for your HF token |

---

## Tested configurations

| Device | iOS | Apple Dev | Backend | Text | Audio (Path A) | Status |
|---|---|---|---|---|---|---|
| iPhone 16 (8 GB RAM) | 18.7.7 | Free | GPU (Metal) | ✅ | ✅ | E2B + E4B both work |

On iPhone 16, **Gemma 4 E4B runs consistently at `maxTokens=1536`** (peak RSS ~3.2 GB). **`maxTokens=2048` exceeds the per-process memory cap** on this device and fails to load reliably — back off to 1536 or lower for E4B. E2B has more headroom and can go higher.

### Raising the memory cap (paid Apple Developer accounts)

If you have a **paid** Apple Developer Program membership, you can attach the increased-memory-limit entitlements to your provisioning profile and push `maxTokens` higher for E4B. The entitlement file is already in this repo at [`ios/Runner/Runner.entitlements`](ios/Runner/Runner.entitlements):

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
<key>com.apple.developer.kernel.increased-debugging-memory-limit</key>
<true/>
```

To enable it: open `ios/Runner.xcworkspace` in Xcode → **Runner** target → **Build Settings** → search for "Code Signing Entitlements" → set the value to `Runner/Runner.entitlements`. Then **Signing & Capabilities** → "+ Capability" → add **Increased Memory Limit**. Rebuild — the new provisioning profile will include the entitlement, and your per-process cap on Gemma-class devices roughly doubles.

⚠️ **Free-tier Apple Developer accounts can't link this entitlement** — Apple's portal will reject the provisioning request. Free tier is hard-capped at the default ~3 GB cap that gates E4B at `maxTokens=2048` on iPhone 16.

If you test on other devices, PRs to update this table are welcome. Decode speed, TTFT, and memory footprint are highly dependent on device, thermal state, maxTokens, and MTP setting — use the in-app stats card to measure your own.

## License

MIT
