import SwiftUI
import BrainBarLifecycle

BrainBarSignalSafety.ignoreSIGPIPE()

#if DEBUG
BrainBarNoEnrichmentRender.runIfRequested()
BrainBarRenderHarness.runIfRequested()
NDRenderHarness.runIfRequested()
#endif

BrainBarApp.main()
