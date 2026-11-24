import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Scan,
  History,
  FileText,
  BrainCircuit,
  User,
  Settings,
  Activity,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/utils/cn";
import { NavItem } from "@/types/navigation";

export interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  className?: string;
  onCloseMobile?: () => void;
}

const PRIMARY_NAV_ITEMS: NavItem[] = [
  {
    title: "Dashboard",
    path: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "X-ray Analysis",
    path: "/analysis",
    icon: Scan,
  },
  {
    title: "Scan History",
    path: "/history",
    icon: History,
  },
  {
    title: "Reports",
    path: "/reports",
    icon: FileText,
  },
  {
    title: "Model Information",
    path: "/model-info",
    icon: BrainCircuit,
  },
];

const ACCOUNT_NAV_ITEMS: NavItem[] = [
  {
    title: "Profile",
    path: "/profile",
    icon: User,
  },
  {
    title: "Settings",
    path: "/settings",
    icon: Settings,
  },
];

export function Sidebar({
  isCollapsed,
  onToggleCollapse,
  className,
  onCloseMobile,
}: SidebarProps) {
  return (
    <aside
      aria-label="Sidebar navigation"
      className={cn(
        "h-full flex flex-col bg-white border-r border-slate-200 transition-all duration-300 ease-in-out motion-reduce:transition-none select-none shadow-sm",
        isCollapsed ? "w-20" : "w-64",
        className,
      )}
    >
      <div className="h-16 px-4 flex items-center justify-between border-b border-slate-100">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-clinical-700 to-clinical-500 text-white flex items-center justify-center shrink-0 shadow-md shadow-clinical-500/20">
            <Activity className="w-5 h-5" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-bold text-slate-900 text-base leading-tight tracking-tight">
                LungCare <span className="text-clinical-600">AI</span>
              </span>
              <span className="text-[10px] uppercase font-semibold tracking-wider text-slate-500">
                Radiology Research
              </span>
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={onToggleCollapse}
          className="hidden lg:flex w-7 h-7 rounded-lg items-center justify-center text-slate-500 hover:text-slate-600 hover:bg-slate-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!isCollapsed}
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        <div className="space-y-1">
          {!isCollapsed && (
            <p className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Clinical Workflow
            </p>
          )}
          {PRIMARY_NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onCloseMobile}
                title={isCollapsed ? item.title : undefined}
                aria-label={item.title}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500",
                    isActive
                      ? "bg-clinical-50 text-clinical-700 font-semibold shadow-xs"
                      : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
                    isCollapsed && "justify-center px-2",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      aria-hidden="true"
                      className={cn(
                        "w-5 h-5 shrink-0 transition-colors",
                        isActive
                          ? "text-clinical-600"
                          : "text-slate-500 group-hover:text-slate-600",
                      )}
                    />
                    <span
                      className={cn(
                        "truncate",
                        isCollapsed ? "sr-only" : "inline",
                      )}
                    >
                      {item.title}
                    </span>
                  </>
                )}
              </NavLink>
            );
          })}
        </div>

        <div className="space-y-1 pt-4 border-t border-slate-100">
          {!isCollapsed && (
            <p className="px-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              System & Account
            </p>
          )}
          {ACCOUNT_NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onCloseMobile}
                title={isCollapsed ? item.title : undefined}
                aria-label={item.title}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500",
                    isActive
                      ? "bg-clinical-50 text-clinical-700 font-semibold shadow-xs"
                      : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
                    isCollapsed && "justify-center px-2",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      aria-hidden="true"
                      className={cn(
                        "w-5 h-5 shrink-0 transition-colors",
                        isActive
                          ? "text-clinical-600"
                          : "text-slate-500 group-hover:text-slate-600",
                      )}
                    />
                    <span
                      className={cn(
                        "truncate",
                        isCollapsed ? "sr-only" : "inline",
                      )}
                    >
                      {item.title}
                    </span>
                  </>
                )}
              </NavLink>
            );
          })}
        </div>
      </div>

      {!isCollapsed && (
        <div className="p-4 m-3 rounded-xl bg-slate-50 border border-slate-200/70">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 mb-1">
            <ShieldCheck className="w-4 h-4 text-clinical-600" />
            <span>DenseNet121 Engine</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Four-class transfer learning model with Grad-CAM explainability.
          </p>
          <div className="mt-2.5 flex flex-col gap-0.5 text-[11px] text-slate-500">
            <span className="text-[10px]">
              Validation Accuracy (Research Dataset)
            </span>
            <span className="font-semibold text-clinical-700 text-xs">
              97.68%
            </span>
          </div>
        </div>
      )}
    </aside>
  );
}
