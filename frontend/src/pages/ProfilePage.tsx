import { User } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

export function ProfilePage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            User Workspace Preferences
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Local browser preferences. Authentication, login credentials, and cloud clinical profiles are not implemented.
          </p>
        </div>
        <Badge variant="outline" size="md">
          Phase 3.8 Target
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardContent className="pt-6 text-center space-y-4">
            <div className="w-20 h-20 rounded-full bg-slate-100 border-2 border-slate-200 text-slate-600 flex items-center justify-center mx-auto text-xl font-bold">
              <User className="w-8 h-8 text-slate-500" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-lg">Research Workspace</h3>
              <p className="text-xs text-slate-500">Guest User (Local Session)</p>
            </div>
            <Badge variant="outline" size="sm">Local Browser Session Only</Badge>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Session Details</CardTitle>
            <CardDescription>
              Local session attributes. No user account data or clinical profiles are stored.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div>
                <label className="font-semibold text-slate-700 block mb-1">Session Mode</label>
                <input
                  type="text"
                  readOnly
                  value="Anonymous / Research Guest"
                  className="w-full px-3 py-2 rounded-lg border border-slate-500 bg-slate-50 text-slate-700 select-none"
                />
              </div>
              <div>
                <label className="font-semibold text-slate-700 block mb-1">Storage Scope</label>
                <input
                  type="text"
                  readOnly
                  value="Browser Local Only (No Remote Account)"
                  className="w-full px-3 py-2 rounded-lg border border-slate-500 bg-slate-50 text-slate-700 select-none"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
