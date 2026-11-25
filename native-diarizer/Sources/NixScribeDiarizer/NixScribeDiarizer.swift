import CoreML
import Foundation
import FluidAudio

private struct Interval: Encodable {
    let start: Float
    let end: Float
    let speaker: String
}

private struct Response: Encodable {
    let intervals: [Interval]
}

private struct Progress: Encodable {
    let phase: String
    let percent: Double
}

@main
struct NixScribeDiarizer {
    static func writeProgress(_ url: URL, _ phase: String, _ percent: Double) {
        try? JSONEncoder().encode(Progress(phase: phase, percent: percent))
            .write(to: url, options: .atomic)
    }

    static func main() async throws {
        let args = CommandLine.arguments
        if args.count == 2 && args[1] == "--self-test" {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            _ = OfflineDiarizerManager(config: OfflineDiarizerConfig())
            print("FluidAudio 0.18.4 helper linked; Core ML configuration available")
            return
        }
        guard args.count == 4 else {
            throw NSError(domain: "NixScribeDiarizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Expected audio path, result path, speaker count or auto"])
        }
        let audio = URL(fileURLWithPath: args[1])
        let output = URL(fileURLWithPath: args[2])
        let progress = output.deletingLastPathComponent().appendingPathComponent("progress.json")
        var config = OfflineDiarizerConfig()
        if args[3] != "auto" {
            guard let count = Int(args[3]), (2...10).contains(count) else {
                throw NSError(domain: "NixScribeDiarizer", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid speaker count"])
            }
            config.clustering.numSpeakers = count
        }
        // Preserve overlap and default fidelity settings. Smaller batches reduce peak memory.
        config.segmentation.batchSize = 8
        config.embedding.batchSize = 8
        let manager = OfflineDiarizerManager(config: config)
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .all
        // Try the standard FluidAudio cache offline first, without purging it.
        writeProgress(progress, "checking", 0)
        ModelHub.offlineMode = true
        let models: OfflineDiarizerModels
        do {
            models = try await OfflineDiarizerModels.load(configuration: modelConfig)
        } catch {
            writeProgress(progress, "preparing", 0)
            ModelHub.offlineMode = false
            models = try await OfflineDiarizerModels.load(configuration: modelConfig)
        }
        // No network downloads during actual audio inference.
        ModelHub.offlineMode = true
        manager.initialize(models: models)
        writeProgress(progress, "inference", 0)
        let result = try await manager.process(audio) { completed, total in
            if total > 0 {
                // Segmentation is only one part of diarization.
                writeProgress(progress, "inference", 70 * Double(completed) / Double(total))
            }
        }
        var ids: [String: String] = [:]
        let rows = result.segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }.map { segment in
            if ids[segment.speakerId] == nil {
                ids[segment.speakerId] = String(format: "SPEAKER_%02d", ids.count)
            }
            return Interval(start: segment.startTimeSeconds, end: segment.endTimeSeconds,
                            speaker: ids[segment.speakerId]!)
        }
        try JSONEncoder().encode(Response(intervals: rows)).write(to: output, options: .atomic)
        writeProgress(progress, "inference", 100)
    }
}
