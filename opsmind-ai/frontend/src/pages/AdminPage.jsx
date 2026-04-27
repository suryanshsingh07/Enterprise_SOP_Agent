import React from 'react';
import UploadPanel from '../components/UploadPanel';

const AdminPage = () => {
    return (
        <div className="w-full h-full p-8 overflow-y-auto" style={{ backgroundImage: 'radial-gradient(circle at top left, rgba(79, 70, 229, 0.1), transparent 40%)' }}>
            <div className="max-w-5xl mx-auto space-y-8">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-100 to-gray-400 bg-clip-text text-transparent">Uploads</h1>
                    <p className="text-gray-400 mt-2">Manage the standard operating procedures that power OpsMind.</p>
                </div>
                
                <UploadPanel />
                
                <div className="glass p-8 rounded-2xl mt-8">
                    <h3 className="text-lg font-semibold mb-4">Indexed Documents</h3>
                    <div className="text-sm text-gray-500 italic">
                        Document listing and deletion to be implemented in a future iteration.
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdminPage;
