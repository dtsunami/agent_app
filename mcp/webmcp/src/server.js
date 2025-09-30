import { spawn } from 'child_process';
import clipboardy from 'clipboardy';
import os from 'os';

export class WebMCPServer {
    constructor(port, headless = false) {
        // Priority: run args → env var → default value
        this.port = port || process.env.WEBMCP_PORT || 8005;
        this.headless = headless || process.env.WEBMCP_HEADLESS === 'true' || false;
        this.hostname = this.getLocalIP();
        this.server = null;
    }

    getLocalIP() {
        const interfaces = os.networkInterfaces();
        for (const name of Object.keys(interfaces)) {
            for (const iface of interfaces[name]) {
                if (iface.family === 'IPv4' && !iface.internal) {
                    return iface.address;
                }
            }
        }
        return '127.0.0.1';
    }

    async start() {
        try {
            // Prepare arguments for @playwright/mcp
            const args = [
                '@playwright/mcp@latest',
                '--port', this.port.toString(),
                '--host', '0.0.0.0'
            ];

            if (this.headless) {
                args.push('--headless');
            }

            console.log(`🎭 Starting Playwright MCP server on port ${this.port}${this.headless ? ' (headless)' : ''}...`);

            // Start the Playwright MCP server
            this.server = spawn('npx', args, {
                stdio: 'inherit',
                shell: true
            });

            // Handle server events
            this.server.on('error', (error) => {
                console.error('❌ Failed to start Playwright MCP server:', error);
                process.exit(1);
            });

            this.server.on('close', (code) => {
                console.log(`Playwright MCP server exited with code ${code}`);
            });

            // Give the server a moment to start
            await new Promise(resolve => setTimeout(resolve, 2000));

            const connectionConfig = {
                name: "webmcp",
                url: `http://${this.hostname}:${this.port}`,
                description: `Browser automation from ${os.hostname()}`,
                tools: [
                    "browser_navigate",
                    "browser_click",
                    "browser_type",
                    "browser_screenshot",
                    "browser_evaluate",
                    "browser_wait_for",
                    "browser_fill_form",
                    "browser_select_option",
                    "browser_hover",
                    "browser_drag",
                    "browser_tabs",
                    "browser_close",
                    "browser_resize",
                    "browser_console_messages",
                    "browser_network_requests",
                    "browser_handle_dialog",
                    "browser_file_upload",
                    "browser_press_key",
                    "browser_take_screenshot",
                    "browser_snapshot",
                    "browser_navigate_back",
                    "browser_install"
                ]
            };

            // Copy connection config to clipboard
            try {
                await clipboardy.write(JSON.stringify(connectionConfig, null, 2));
                console.log('✅ Connection command copied to clipboard!');
            } catch (clipError) {
                console.log('⚠️ Could not copy to clipboard:', clipError.message);
            }

            console.log('📋 Browser MCP Connection Config:');
            console.log(JSON.stringify(connectionConfig, null, 2));
            console.log('');
            console.log(`🌐 WebMCP server running on ${this.hostname}:${this.port}`);
            console.log('🎭 Playwright browser automation ready');
            console.log('📋 Connection config copied to clipboard');
            console.log('');
            console.log('Ready for da_code connection!');
            console.log('> add_mcp ' + JSON.stringify(connectionConfig));

        } catch (error) {
            console.error('❌ Failed to start WebMCP server:', error);
            process.exit(1);
        }
    }

    async stop() {
        if (this.server) {
            this.server.kill('SIGTERM');
            console.log('WebMCP server stopped');
        }
    }
}

// Handle process termination
process.on('SIGINT', async () => {
    console.log('\nShutting down WebMCP server...');
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\nShutting down WebMCP server...');
    process.exit(0);
});