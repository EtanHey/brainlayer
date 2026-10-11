import SwiftUI
import BrainBarLifecycle

BrainBarSignalSafety.ignoreSIGPIPE()

#if DEBUG
BrainBarNoEnrichmentRender.runIfRequested()
BrainBarRenderHarness.runIfRequested()
#endif

BrainBarApp.main()
