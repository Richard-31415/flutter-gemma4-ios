// Minimal Flutter example: running Gemma 4 on iOS via flutter_gemma.
//
// Tested working on iPhone 16 (iOS 18.7.7), free Apple Developer account,
// with the "Increased Memory Limit" entitlement applied via GetMoreRam
// (https://github.com/hugeBlack/GetMoreRam). See README for details.

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
// ModelThinkingFilter isn't in the top-level export — imported directly.
// It parses Gemma 4's <|channel>thought\n...<channel|> stream into
// separate ThinkingResponse / TextResponse events.
import 'package:flutter_gemma/core/extensions.dart' show ModelThinkingFilter;

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

  // The literal `<|think|>` control token at the start of the system prompt
  // enables Gemma 4's thinking mode. Remove it to disable.
  final _systemPromptCtrl = TextEditingController(
    text: '<|think|>You are a helpful assistant. Think step by step before answering.',
  );
  final _promptCtrl = TextEditingController(
    text: 'In 2-3 sentences, explain why the sky is blue.',
  );

  String _thinking = '';
  String _answer = '';
  bool _generating = false;
  StreamSubscription<ModelResponse>? _sub;

  @override
  void dispose() {
    _sub?.cancel();
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
      // GOTCHA: flutter_gemma defaults fileType to ModelFileType.task. For
      // .litertlm files (Gemma 4), you MUST pass ModelFileType.litertlm
      // explicitly. Otherwise the chat-template router on iOS skips manual
      // turn-marker formatting → prefill crashes with a NULL memset in
      // LlmLiteRTExecutor::PrefillInternal.
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
      // Pre-warm the model so memory errors surface here, not at inference time.
      final _ = await FlutterGemma.getActiveModel(
        maxTokens: 2048,
        preferredBackend: PreferredBackend.gpu,
      );
      setState(() {
        _modelReady = true;
        _status = 'Model ready.';
      });
    } catch (e) {
      setState(() => _status = 'Download/load failed: $e');
    } finally {
      setState(() => _downloading = false);
    }
  }

  Future<void> _run() async {
    setState(() {
      _generating = true;
      _thinking = '';
      _answer = '';
      _status = 'Running inference…';
    });

    try {
      final model = await FlutterGemma.getActiveModel(
        maxTokens: 2048,
        preferredBackend: PreferredBackend.gpu,
      );
      final chat = await model.createChat(
        systemInstruction: _systemPromptCtrl.text.trim().isEmpty
            ? null
            : _systemPromptCtrl.text.trim(),
      );
      await chat.addQueryChunk(
        Message.text(text: _promptCtrl.text, isUser: true),
      );

      // GOTCHA: flutter_gemma 0.15.0 does NOT auto-route Gemma 4's thinking
      // tokens into ThinkingResponse events — you have to wrap the stream
      // with ModelThinkingFilter.filterThinkingStream() manually. Without
      // this, the raw `<|channel>thought\n...<channel|>` markers dump into
      // the answer text.
      final rawStream = chat.generateChatResponseAsync();
      final filteredStream = ModelThinkingFilter.filterThinkingStream(
        rawStream,
        modelType: _choice.modelType,
      );
      _sub = filteredStream.listen(
        (response) {
          if (!mounted) return;
          if (response is ThinkingResponse) {
            setState(() => _thinking += response.content);
          } else if (response is TextResponse) {
            setState(() => _answer += response.token);
          }
        },
        onDone: () {
          if (!mounted) return;
          setState(() {
            _generating = false;
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
      ),
      body: Padding(
        padding: const EdgeInsets.all(12),
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
                    FilledButton.icon(
                      onPressed: _modelReady && !_generating ? _run : null,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Run inference'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // Output
            Expanded(
              child: Card(
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
                                    final text =
                                        'PROMPT:\n${_promptCtrl.text}\n\n'
                                        'SYSTEM:\n${_systemPromptCtrl.text}\n\n'
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
                      Expanded(
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
                      Expanded(
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
            ),
          ],
        ),
      ),
    );
  }
}
