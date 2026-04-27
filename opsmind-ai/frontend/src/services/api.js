export const uploadSOP = async (file) => {
    const formData = new FormData();
    formData.append('document', file);
    
    const apiKey = localStorage.getItem('opsmind_gemini_key') || '';

    const response = await fetch('/api/admin/upload', {
        method: 'POST',
        headers: { 'x-gemini-api-key': apiKey },
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Upload failed');
    }
    
    return response.json();
};

export const getDocuments = async () => {
    const response = await fetch('/api/admin/documents');
    if (!response.ok) {
        throw new Error('Failed to fetch documents');
    }
    return response.json();
};
