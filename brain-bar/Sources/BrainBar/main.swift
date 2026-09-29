import SwiftUI
import BrainBarLifecycle

BrainBarSignalSafety.ignoreSIGPIPE()

#if DEBUG
BrainBarRenderHarness.runIfRequested()
#endif

BrainBarApp.main()
