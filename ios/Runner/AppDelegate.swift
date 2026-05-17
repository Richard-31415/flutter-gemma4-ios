import Flutter
import UIKit
import Darwin

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    // phys_footprint is what iOS's per-process memory cap actually enforces,
    // and what Xcode's Debug Navigator displays. It includes Metal/IOKit
    // allocations (where Gemma's GPU weights live), unlike resident_size
    // which dart:io's ProcessInfo.currentRss reports.
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "MemoryStatsPlugin") {
      let channel = FlutterMethodChannel(
        name: "gemma4_demo/memory_stats",
        binaryMessenger: registrar.messenger()
      )
      channel.setMethodCallHandler { call, result in
        switch call.method {
        case "physFootprint":
          result(NSNumber(value: physFootprintBytes()))
        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }
  }
}

private func physFootprintBytes() -> Int64 {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) /
              mach_msg_type_number_t(MemoryLayout<integer_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
    ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  return kr == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
}
