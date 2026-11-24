import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { DisclaimerBanner } from './DisclaimerBanner';
import { useFocusTrap } from '@/hooks/useFocusTrap';

const ROUTE_TITLES: Record<string, { title: string; context: string }> = {
  '/dashboard': { title: 'Dashboard', context: 'Overview & Activity' },
  '/analysis': { title: 'X-ray Analysis', context: 'Radiology Computer Vision' },
  '/history': { title: 'Scan History', context: 'Session Records' },
  '/reports': { title: 'Clinical Reports', context: 'Documentation & Synthesis' },
  '/model-info': { title: 'Model Architecture', context: 'DenseNet121 & Explainability' },
  '/profile': { title: 'Researcher Profile', context: 'User Preferences' },
  '/settings': { title: 'System Settings', context: 'Configuration & Telemetry' },
};

export function AppLayout() {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [isMobileOpen, setIsMobileOpen] = React.useState(false);

  const drawerRef = useFocusTrap<HTMLDivElement>({
    isActive: isMobileOpen,
    onClose: () => setIsMobileOpen(false),
  });

  const currentRouteInfo = ROUTE_TITLES[location.pathname] || {
    title: 'Research Console',
    context: 'LungCare AI',
  };

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block shrink-0 sticky top-0 h-screen z-30">
        <Sidebar
          isCollapsed={isCollapsed}
          onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        />
      </div>

      {/* Mobile Drawer Backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 lg:hidden transition-opacity motion-reduce:transition-none"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Mobile Drawer Content */}
      <div
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-label="Mobile Navigation"
        aria-hidden={!isMobileOpen}
        tabIndex={-1}
        className={`fixed inset-y-0 left-0 w-64 z-50 lg:hidden transform transition-transform duration-300 ease-in-out motion-reduce:transition-none ${
          isMobileOpen
            ? 'translate-x-0 visible'
            : '-translate-x-full invisible pointer-events-none'
        }`}
      >
        <Sidebar
          isCollapsed={false}
          onToggleCollapse={() => {}}
          onCloseMobile={() => setIsMobileOpen(false)}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        {/* Global Clinical Disclaimer Banner */}
        <DisclaimerBanner />

        {/* Global Header */}
        <Header
          pageTitle={currentRouteInfo.title}
          contextLabel={currentRouteInfo.context}
          onOpenMobileMenu={() => setIsMobileOpen(true)}
        />

        {/* Page Content Outlet */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          <Outlet />
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200/80 bg-white py-4 px-6 text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              LungCare AI &copy; {new Date().getFullYear()} — Academic & Research Prototype.
            </span>
            <span className="text-slate-500">
              DenseNet121 • Grad-CAM • FastAPI Backend
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
