export const streamQueryResponse = async (query, apiKey, onUpdate, onComplete, onError) => {
    try {
        const response = await fetch('/api/query/stream', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'x-gemini-api-key': apiKey || ''
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error('Stream failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let sources = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.replace('data: ', '');
                    if (!dataStr) continue;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.done) {
                            onComplete(fullText, sources);
                            return;
                        }
                        if (data.error) {
                            onError(data.error);
                            return;
                        }
                        
                        if (data.text) {
                            if (data.text.includes('__SOURCES__')) {
                                const parts = data.text.split('__SOURCES__');
                                fullText += parts[0];
                                try {
                                    sources = JSON.parse(parts[1]);
                                } catch(e) {}
                            } else {
                                fullText += data.text;
                            }
                            onUpdate(fullText, sources);
                        }
                    } catch (e) {
                        console.error("Parse error in stream", e);
                    }
                }
            }
        }
    } catch (err) {
        onError(err.message);
    }
};
