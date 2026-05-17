// Minimal Flutter example: running Gemma 4 on iOS via flutter_gemma.
//
// Demonstrates: model download, GPU inference, native audio input
// (Message.withAudio for Gemma 4 E2B/E4B), thinking-mode parsing,
// and in-app benchmarking (TTFT, tok/s, memory).
//
// Tested working on iPhone 16 (iOS 18.7.7), free Apple Developer account,
// with the "Increased Memory Limit" entitlement applied via GetMoreRam
// (https://github.com/hugeBlack/GetMoreRam). See README for details.

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
// ModelThinkingFilter isn't in the top-level export — imported directly.
// It parses Gemma 4's <|channel>thought\n...<channel|> stream into
// separate ThinkingResponse / TextResponse events.
import 'package:flutter_gemma/core/extensions.dart' show ModelThinkingFilter;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // HF token is injected at build time via `--dart-define-from-file=config.json`.
  // See config.json.example for the expected format. Token only needed for
  // gated repos; Gemma 4 on litert-community is currently public.
  const token = String.fromEnvironment('HUGGINGFACE_TOKEN');
  FlutterGemma.initialize(
    huggingFaceToken: token.isNotEmpty ? token : null,
    maxDownloadRetries: 10,
  );

  runApp(const Gemma4IosDemo());
}

class Gemma4IosDemo extends StatelessWidget {
  const Gemma4IosDemo({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma 4 iOS Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class ModelChoice {
  final String label;
  final String url;
  final ModelType modelType;
  final int approxSizeMb;
  const ModelChoice(this.label, this.url, this.modelType, this.approxSizeMb);
}

const _models = <ModelChoice>[
  ModelChoice(
    'Gemma 4 E2B (~2.4 GB)',
    'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
    ModelType.gemma4,
    2400,
  ),
  ModelChoice(
    'Gemma 4 E4B (~4.3 GB)',
    'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm',
    ModelType.gemma4,
    4300,
  ),
];

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  ModelChoice _choice = _models.first;
  int _downloadProgress = 0;
  bool _downloading = false;
  bool _modelReady = false;
  String _status = 'Idle.';

  // Engine settings. These are baked in at model load time — once the model
  // is loaded, both controls are disabled. To change them, hot-restart (R)
  // and load again.
  int _maxTokens = 1024;
  // null = use the .litertlm file's default (MTP on for litert-community
  // Gemma 4 E2B/E4B). true/false forces it explicitly.
  bool? _speculativeDecoding;

  // The literal `<|think|>` control token at the start of the system prompt
  // enables Gemma 4's thinking mode. Remove it to disable.
  final _systemPromptCtrl = TextEditingController(
    text:
        '<|think|>You are a helpful assistant. Think step by step before answering.',
  );
  final _promptCtrl = TextEditingController(
    text: 'In 2-3 sentences, explain why the sky is blue.',
  );

  String _thinking = '';
  String _answer = '';
  bool _generating = false;
  StreamSubscription<ModelResponse>? _sub;

  // Audio (Path A: native Gemma 4 audio via Message.withAudio)
  final AudioRecorder _recorder = AudioRecorder();
  bool _recording = false;
  final Stopwatch _recordSw = Stopwatch();
  Timer? _recordTicker;
  String _recordStatus = '';

  // Benchmarking
  DateTime? _runStart;
  int? _ttftMs;            // ms from Run press → first stream event
  int _thinkingTokens = 0; // stream events of type ThinkingResponse
  int _answerTokens = 0;   // stream events of type TextResponse
  int? _totalMs;           // ms from Run press → onDone
  double? _decodeTokPerSec;
  int? _rssIdleMb;         // memory snapshot after model load, before run
  int? _rssPeakMb;         // memory snapshot at end of run

  // iOS's per-process memory cap is enforced against phys_footprint, not
  // resident_size — and Gemma's GPU weights live in Metal/IOKit memory which
  // doesn't show up in ProcessInfo.currentRss. Read phys_footprint via a
  // Swift platform channel (see ios/Runner/AppDelegate.swift). Falls back to
  // currentRss on non-iOS or if the channel call fails.
  static const _memChannel = MethodChannel('gemma4_demo/memory_stats');
  Future<int> _currentRssMb() async {
    try {
      final n = await _memChannel.invokeMethod<num>('physFootprint');
      if (n != null && n > 0) return (n / 1024 / 1024).round();
    } catch (_) {}
    return (ProcessInfo.currentRss / 1024 / 1024).round();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _recorder.dispose();
    _recordTicker?.cancel();
    _systemPromptCtrl.dispose();
    _promptCtrl.dispose();
    super.dispose();
  }

  Future<void> _download() async {
    setState(() {
      _downloading = true;
      _downloadProgress = 0;
      _status = 'Downloading ${_choice.label}…';
    });
    try {
      await FlutterGemma.installModel(
            modelType: _choice.modelType,
            fileType: ModelFileType.litertlm,
          )
          .fromNetwork(
            _choice.url,
            token: const String.fromEnvironment('HUGGINGFACE_TOKEN'),
          )
          .withProgress((p) {
            if (mounted) setState(() => _downloadProgress = p);
          })
          .install();
      setState(() {
        _status = 'Model installed. Loading into memory…';
      });
      // Optional pre-warm — getActiveModel loads into memory lazily on first use,
      // but pulling it here surfaces memory issues immediately rather than at inference time.
      // supportAudio: true at the ENGINE level loads Gemma 4's ~300M audio encoder
      // weights. Without this, sending Message.withAudio later NULL-derefs in
      // ProcessAndCombineContents because the audio encoder pointer is unset.
      final _ = await FlutterGemma.getActiveModel(
        maxTokens: _maxTokens,
        preferredBackend: PreferredBackend.gpu,
        supportAudio: true,
        enableSpeculativeDecoding: _speculativeDecoding,
      );
      final idleRss = await _currentRssMb();
      if (!mounted) return;
      setState(() {
        _modelReady = true;
        _rssIdleMb = idleRss;
        _status = 'Model ready. RSS idle: ${_rssIdleMb} MB';
      });
    } catch (e) {
      setState(() => _status = 'Download/load failed: $e');
    } finally {
      setState(() => _downloading = false);
    }
  }

  Future<void> _toggleRecord() async {
    if (_recording) {
      await _stopAndRunWithAudio();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (!await _recorder.hasPermission()) {
      setState(() => _recordStatus = 'Mic permission denied.');
      return;
    }
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/gemma4_demo_audio.wav';
    // 16 kHz mono WAV — standard ASR format MediaPipe / LiteRT-LM expects.
    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: path,
    );
    _recordSw
      ..reset()
      ..start();
    _recordTicker = Timer.periodic(const Duration(milliseconds: 200), (_) {
      if (mounted) setState(() {});
    });
    setState(() {
      _recording = true;
      _recordStatus = '';
    });
  }

  Future<void> _stopAndRunWithAudio() async {
    _recordSw.stop();
    _recordTicker?.cancel();
    final path = await _recorder.stop();
    setState(() {
      _recording = false;
    });
    if (path == null) {
      setState(() => _recordStatus = 'No audio captured.');
      return;
    }
    final bytes = await File(path).readAsBytes();
    setState(() {
      _recordStatus =
          'Captured ${(bytes.length / 1024).toStringAsFixed(1)} KB '
          '(${_recordSw.elapsed.inMilliseconds / 1000}s). Running…';
    });
    await _run(audioBytes: bytes);
  }

  Future<void> _run({Uint8List? audioBytes}) async {
    setState(() {
      _generating = true;
      _thinking = '';
      _answer = '';
      _runStart = DateTime.now();
      _ttftMs = null;
      _thinkingTokens = 0;
      _answerTokens = 0;
      _totalMs = null;
      _decodeTokPerSec = null;
      _rssPeakMb = null;
      _status = audioBytes == null
          ? 'Running inference…'
          : 'Running inference with audio (${audioBytes.length} bytes)…';
    });

    try {
      final model = await FlutterGemma.getActiveModel(
        maxTokens: _maxTokens,
        preferredBackend: PreferredBackend.gpu,
        supportAudio: true,
        enableSpeculativeDecoding: _speculativeDecoding,
      );
      final chat = await model.createChat(
        systemInstruction: _systemPromptCtrl.text.trim().isEmpty
            ? null
            : _systemPromptCtrl.text.trim(),
        supportAudio: audioBytes != null,
      );
      if (audioBytes != null) {
        await chat.addQueryChunk(
          Message.withAudio(
            text:
                'Listen to the audio and respond following the system instruction.',
            audioBytes: audioBytes,
            isUser: true,
          ),
        );
      } else {
        await chat.addQueryChunk(
          Message.text(text: _promptCtrl.text, isUser: true),
        );
      }

      final rawStream = chat.generateChatResponseAsync();
      final filteredStream = ModelThinkingFilter.filterThinkingStream(
        rawStream,
        modelType: _choice.modelType,
      );
      _sub = filteredStream.listen(
        (response) {
          if (!mounted) return;
          final elapsed =
              DateTime.now().difference(_runStart!).inMilliseconds;
          _ttftMs ??= elapsed;
          if (response is ThinkingResponse) {
            setState(() {
              _thinking += response.content;
              _thinkingTokens++;
            });
          } else if (response is TextResponse) {
            setState(() {
              _answer += response.token;
              _answerTokens++;
            });
          }
        },
        onDone: () async {
          if (!mounted) return;
          final total =
              DateTime.now().difference(_runStart!).inMilliseconds;
          final totalToks = _thinkingTokens + _answerTokens;
          final decodeMs = (_ttftMs == null) ? total : (total - _ttftMs!);
          final speed = (decodeMs > 0 && totalToks > 1)
              ? ((totalToks - 1) / (decodeMs / 1000.0))
              : null;
          final endRss = await _currentRssMb();
          if (!mounted) return;
          setState(() {
            _generating = false;
            _totalMs = total;
            _decodeTokPerSec = speed;
            _rssPeakMb = endRss;
            _status = 'Done.';
          });
        },
        onError: (e) {
          if (!mounted) return;
          setState(() {
            _generating = false;
            _status = 'Inference error: $e';
          });
        },
      );
    } catch (e) {
      setState(() {
        _generating = false;
        _status = 'Setup error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 4 iOS Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        toolbarHeight: 40,
      ),
      // Tap anywhere outside a TextField to dismiss the keyboard.
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () => FocusScope.of(context).unfocus(),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
            // Model setup
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text('Model', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    DropdownButton<ModelChoice>(
                      isExpanded: true,
                      value: _choice,
                      onChanged: _downloading || _modelReady
                          ? null
                          : (c) => setState(() => _choice = c!),
                      items: _models
                          .map((m) => DropdownMenuItem(value: m, child: Text(m.label)))
                          .toList(),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('maxTokens',
                                  style: TextStyle(fontSize: 11, color: Colors.grey)),
                              DropdownButton<int>(
                                isExpanded: true,
                                value: _maxTokens,
                                onChanged: _downloading || _modelReady
                                    ? null
                                    : (v) => setState(() => _maxTokens = v!),
                                items: const [512, 1024, 1536, 1792, 2048]
                                    .map((v) => DropdownMenuItem(
                                          value: v,
                                          child: Text('$v'),
                                        ))
                                    .toList(),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('MTP (speculative decoding)',
                                  style: TextStyle(fontSize: 11, color: Colors.grey)),
                              DropdownButton<int>(
                                isExpanded: true,
                                // 0 = null (model default), 1 = true, 2 = false
                                value: _speculativeDecoding == null
                                    ? 0
                                    : (_speculativeDecoding! ? 1 : 2),
                                onChanged: _downloading || _modelReady
                                    ? null
                                    : (v) => setState(() {
                                          _speculativeDecoding = switch (v) {
                                            1 => true,
                                            2 => false,
                                            _ => null,
                                          };
                                        }),
                                items: const [
                                  DropdownMenuItem(value: 0, child: Text('Default (model)')),
                                  DropdownMenuItem(value: 1, child: Text('On')),
                                  DropdownMenuItem(value: 2, child: Text('Off')),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    if (_modelReady)
                      const Padding(
                        padding: EdgeInsets.only(top: 2),
                        child: Text(
                          'Settings locked. Kill and relaunch the app to change '
                          '— the native model can\'t be unloaded in-process.',
                          style: TextStyle(fontSize: 11, color: Colors.grey),
                        ),
                      ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        FilledButton.icon(
                          onPressed: _downloading || _modelReady ? null : _download,
                          icon: const Icon(Icons.download),
                          label: const Text('Download + load'),
                        ),
                        const SizedBox(width: 12),
                        if (_downloading)
                          Expanded(
                            child: LinearProgressIndicator(
                              value: _downloadProgress / 100.0,
                            ),
                          ),
                        if (_downloading) const SizedBox(width: 8),
                        if (_downloading) Text('$_downloadProgress%'),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(_status, style: const TextStyle(color: Colors.grey)),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // Inputs
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text('System instruction',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    TextField(controller: _systemPromptCtrl, maxLines: 2),
                    const SizedBox(height: 8),
                    const Text('Prompt', style: TextStyle(fontWeight: FontWeight.bold)),
                    TextField(controller: _promptCtrl, maxLines: 3),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        FilledButton.icon(
                          onPressed: _modelReady && !_generating && !_recording
                              ? () => _run()
                              : null,
                          icon: const Icon(Icons.play_arrow),
                          label: const Text('Run (text)'),
                        ),
                        const SizedBox(width: 12),
                        FilledButton.icon(
                          onPressed: _modelReady && !_generating
                              ? _toggleRecord
                              : null,
                          icon: Icon(_recording ? Icons.stop : Icons.mic),
                          label: Text(_recording
                              ? 'Stop (${_recordSw.elapsed.inSeconds}s) & run audio'
                              : 'Record + run audio'),
                          style: _recording
                              ? FilledButton.styleFrom(
                                  backgroundColor: Colors.red,
                                )
                              : null,
                        ),
                      ],
                    ),
                    if (_recordStatus.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(_recordStatus,
                            style: const TextStyle(color: Colors.grey)),
                      ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // Stats
            Card(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 8),
                child: Wrap(
                  spacing: 16,
                  runSpacing: 4,
                  children: [
                    _stat(
                        'TTFT',
                        _ttftMs == null
                            ? '—'
                            : '${_ttftMs} ms'),
                    _stat(
                        'Total',
                        _totalMs == null
                            ? '—'
                            : '${(_totalMs! / 1000).toStringAsFixed(2)} s'),
                    _stat(
                        'Speed',
                        _decodeTokPerSec == null
                            ? '—'
                            : '${_decodeTokPerSec!.toStringAsFixed(1)} tok/s'),
                    _stat(
                        'Tokens',
                        '${_thinkingTokens + _answerTokens}  '
                        '(${_thinkingTokens} t · ${_answerTokens} a)'),
                    _stat(
                        'RSS idle',
                        _rssIdleMb == null ? '—' : '${_rssIdleMb} MB'),
                    _stat(
                        'RSS peak',
                        _rssPeakMb == null
                            ? '—'
                            : '${_rssPeakMb} MB '
                              '(+${(_rssPeakMb! - (_rssIdleMb ?? 0))})'),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // Output
            Card(
              child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('Thinking',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          TextButton.icon(
                            onPressed: (_thinking.isEmpty && _answer.isEmpty)
                                ? null
                                : () {
                                    final mtp = _speculativeDecoding == null
                                        ? 'default'
                                        : (_speculativeDecoding! ? 'on' : 'off');
                                    final stats =
                                        'CONFIG:\n'
                                        '  Model: ${_choice.label}\n'
                                        '  maxTokens: $_maxTokens\n'
                                        '  MTP: $mtp\n'
                                        'STATS:\n'
                                        '  TTFT: ${_ttftMs ?? "—"} ms\n'
                                        '  Total: ${_totalMs == null ? "—" : "${(_totalMs! / 1000).toStringAsFixed(2)} s"}\n'
                                        '  Speed: ${_decodeTokPerSec == null ? "—" : "${_decodeTokPerSec!.toStringAsFixed(1)} tok/s"}\n'
                                        '  Tokens: ${_thinkingTokens + _answerTokens} '
                                        '(${_thinkingTokens} thinking · ${_answerTokens} answer)\n'
                                        '  RSS idle: ${_rssIdleMb ?? "—"} MB\n'
                                        '  RSS peak: ${_rssPeakMb ?? "—"} MB';
                                    final text =
                                        'PROMPT:\n${_promptCtrl.text}\n\n'
                                        'SYSTEM:\n${_systemPromptCtrl.text}\n\n'
                                        '$stats\n\n'
                                        'THINKING:\n$_thinking\n\n'
                                        'ANSWER:\n$_answer';
                                    Clipboard.setData(
                                        ClipboardData(text: text));
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(
                                        content: Text('Copied to clipboard'),
                                        duration: Duration(seconds: 1),
                                      ),
                                    );
                                  },
                            icon: const Icon(Icons.copy, size: 16),
                            label: const Text('Copy all'),
                          ),
                        ],
                      ),
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxHeight: 240),
                        child: SingleChildScrollView(
                          child: SelectableText(
                            _thinking.isEmpty ? '(no thinking emitted yet)' : _thinking,
                            style: const TextStyle(
                                fontFamily: 'Menlo', color: Colors.deepPurple),
                          ),
                        ),
                      ),
                      const Divider(),
                      const Text('Answer',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxHeight: 240),
                        child: SingleChildScrollView(
                          child: SelectableText(
                            _answer.isEmpty ? '(awaiting response)' : _answer,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _stat(String label, String value) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(fontSize: 10, color: Colors.grey)),
        Text(value,
            style: const TextStyle(
                fontFamily: 'Menlo',
                fontSize: 13,
                fontWeight: FontWeight.w600)),
      ],
    );
  }
}
