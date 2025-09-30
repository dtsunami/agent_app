import express from 'express';
import cors from 'cors';
import clipboardy from 'clipboardy';
import os from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class ClipjsServer {
    constructor(port = 3000) {
        this.port = port;
        this.app = express();
        this.setupMiddleware();
        this.setupRoutes();

        this.tools = {
            read_text: {
                name: "read_text",
                description: "Read text content from Windows clipboard",
                inputSchema: {
                    type: "object",
                    properties: {},
                    additionalProperties: false
                }
            },
            read_image: {
                name: "read_image",
                description: "Read image from Windows clipboard and return as base64",
                inputSchema: {
                    type: "object",
                    properties: {
                        format: {
                            type: "string",
                            enum: ["PNG", "JPEG"],
                            default: "PNG"
                        }
                    },
                    additionalProperties: false
                }
            },
            write_text: {
                name: "write_text",
                description: "Write text content to Windows clipboard",
                inputSchema: {
                    type: "object",
                    properties: {
                        text: {
                            type: "string",
                            description: "Text to write to clipboard"
                        }
                    },
                    required: ["text"],
                    additionalProperties: false
                }
            },
            write_image: {
                name: "write_image",
                description: "Write base64 image to Windows clipboard",
                inputSchema: {
                    type: "object",
                    properties: {
                        image_data: {
                            type: "string",
                            description: "Base64 encoded image data"
                        },
                        format: {
                            type: "string",
                            enum: ["PNG", "JPEG"],
                            default: "PNG"
                        }
                    },
                    required: ["image_data"],
                    additionalProperties: false
                }
            }
        };
    }

    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json());
    }

    setupRoutes() {
        // Root endpoint
        this.app.get('/', (req, res) => {
            res.json({
                name: "ClipJS - Node.js Clipboard MCP Server",
                version: "1.0.0",
                status: "running",
                tools_available: Object.keys(this.tools).length,
                connection_prompt: this.generateConnectionPrompt()
            });
        });

        // MCP protocol endpoints
        this.app.get('/mcp/tools', (req, res) => {
            res.json({ tools: Object.values(this.tools) });
        });

        this.app.post('/mcp/call/:toolName', async (req, res) => {
            const { toolName } = req.params;
            const { arguments: args = {} } = req.body;

            if (!this.tools[toolName]) {
                return res.status(404).json({
                    content: [{ type: "text", text: `Tool '${toolName}' not found` }],
                    isError: true
                });
            }

            try {
                let result;
                if (toolName === 'read_text') {
                    result = await this.readClipboardText();
                } else if (toolName === 'read_image') {
                    const format = args.format || 'PNG';
                    result = await this.readClipboardImage(format);
                } else if (toolName === 'write_text') {
                    const text = args.text;
                    if (!text) {
                        result = "❌ Error: 'text' parameter is required";
                    } else {
                        result = await this.writeClipboardText(text);
                    }
                } else if (toolName === 'write_image') {
                    const imageData = args.image_data;
                    const format = args.format || 'PNG';
                    if (!imageData) {
                        result = "❌ Error: 'image_data' parameter is required";
                    } else {
                        result = await this.writeClipboardImage(imageData, format);
                    }
                } else {
                    throw new Error(`Unknown tool: ${toolName}`);
                }

                res.json({
                    content: [{ type: "text", text: result }],
                    isError: false
                });
            } catch (error) {
                res.json({
                    content: [{ type: "text", text: `Error: ${error.message}` }],
                    isError: true
                });
            }
        });

        this.app.get('/mcp/connect', (req, res) => {
            res.json({ prompt: this.generateConnectionPrompt() });
        });
    }

    getLocalIP() {
        const interfaces = os.networkInterfaces();
        for (const name of Object.keys(interfaces)) {
            for (const interface of interfaces[name]) {
                if (interface.family === 'IPv4' && !interface.internal) {
                    return interface.address;
                }
            }
        }
        return 'localhost';
    }

    generateConnectionPrompt() {
        const ip = this.getLocalIP();
        const config = {
            name: "clipboard",
            url: `http://${ip}:${this.port}`,
            port: this.port,
            description: `Windows clipboard MCP server at ${ip}`,
            tools: Object.keys(this.tools)
        };
        // Return compact JSON as complete command
        return `add_mcp ${JSON.stringify(config)}`;
    }

    async readClipboardText() {
        try {
            const text = await clipboardy.read();

            if (!text) {
                return "📋 Clipboard is empty or contains no text";
            }

            if (text.length > 10000) {
                return `📋 **Clipboard Text** (truncated from ${text.length.toLocaleString()} chars):\n\n${text.substring(0, 10000)}...\n\n*(truncated)*`;
            }

            return `📋 **Clipboard Text:**\n\n${text}`;
        } catch (error) {
            return `❌ Error reading clipboard text: ${error.message}`;
        }
    }

    async readClipboardImage(format = 'PNG') {
        try {
            // For Windows, try PowerShell to get clipboard image
            if (process.platform === 'win32') {
                const psScript = `
                    Add-Type -AssemblyName System.Windows.Forms
                    $clip = [System.Windows.Forms.Clipboard]::GetImage()
                    if ($clip -ne $null) {
                        $ms = New-Object System.IO.MemoryStream
                        $clip.Save($ms, [System.Drawing.Imaging.ImageFormat]::${format})
                        $bytes = $ms.ToArray()
                        [System.Convert]::ToBase64String($bytes)
                    } else {
                        "NO_IMAGE"
                    }
                `;

                const { stdout } = await execAsync(`powershell -Command "${psScript}"`);
                const base64Data = stdout.trim();

                if (base64Data === 'NO_IMAGE') {
                    return "📋 No image found in clipboard. Copy an image first.";
                }

                // Calculate approximate dimensions and size
                const buffer = Buffer.from(base64Data, 'base64');
                const sizeBytes = buffer.length;

                const result = `🖼️ **Image from Clipboard:**\n` +
                    `**Format:** ${format}\n` +
                    `**Size:** ${sizeBytes.toLocaleString()} bytes\n` +
                    `**Base64 Length:** ${base64Data.length.toLocaleString()} characters\n\n` +
                    `**Base64 Data:**\n${base64Data}\n\n` +
                    `✅ Image successfully read from clipboard`;

                return result;
            } else {
                return "❌ Image clipboard reading is only supported on Windows";
            }
        } catch (error) {
            return `❌ Error reading clipboard image: ${error.message}`;
        }
    }

    async writeClipboardText(text) {
        try {
            await clipboardy.write(text);
            const charCount = text.length;
            return `✅ **Text Written to Clipboard:**\n\n${charCount.toLocaleString()} characters written successfully`;
        } catch (error) {
            return `❌ Error writing clipboard text: ${error.message}`;
        }
    }

    async writeClipboardImage(imageData, format = 'PNG') {
        try {
            if (process.platform === 'win32') {
                const fs = require('fs');
                const path = require('path');
                const os = require('os');

                // Decode base64 image
                let imageBuffer;
                try {
                    imageBuffer = Buffer.from(imageData, 'base64');
                } catch (error) {
                    return `❌ Invalid base64 image data: ${error.message}`;
                }

                // Create temporary file
                const tempDir = os.tmpdir();
                const tempFile = path.join(tempDir, `clipboard_image_${Date.now()}.${format.toLowerCase()}`);

                try {
                    fs.writeFileSync(tempFile, imageBuffer);

                    // Use PowerShell to copy image to clipboard
                    const psScript = `
                        Add-Type -AssemblyName System.Windows.Forms
                        $image = [System.Drawing.Image]::FromFile("${tempFile.replace(/\\/g, '\\\\')}")
                        [System.Windows.Forms.Clipboard]::SetImage($image)
                        $image.Dispose()
                    `;

                    const { exec } = require('child_process');
                    const { promisify } = require('util');
                    const execAsync = promisify(exec);

                    await execAsync(`powershell -Command "${psScript}"`);

                    // Clean up temp file
                    fs.unlinkSync(tempFile);

                    const sizeKB = Math.round(imageBuffer.length / 1024);
                    return `✅ **Image Written to Clipboard:**\n\n${format} image (${sizeKB} KB) written successfully`;

                } catch (error) {
                    // Clean up temp file if it exists
                    if (fs.existsSync(tempFile)) {
                        fs.unlinkSync(tempFile);
                    }
                    return `❌ Error copying image to clipboard: ${error.message}`;
                }
            } else {
                return "❌ Image clipboard writing is only supported on Windows";
            }
        } catch (error) {
            return `❌ Error writing clipboard image: ${error.message}`;
        }
    }

    async copyConnectionPromptToClipboard() {
        try {
            const command = this.generateConnectionPrompt();
            await clipboardy.write(command);
            console.log(`✅ Complete command copied to clipboard:`);
            console.log(`   ${command}`);
        } catch (error) {
            console.log(`❌ Could not copy to clipboard: ${error.message}`);
        }
    }

    async start() {
        console.log(`\n📎 Starting ClipJS (Node.js Clipboard) MCP Server on port ${this.port}`);
        console.log(`📋 Local access: http://localhost:${this.port}`);
        console.log(`🌐 Network access: http://${this.getLocalIP()}:${this.port}`);
        console.log(`🔧 Tools: ${Object.keys(this.tools).join(', ')}`);

        // Copy connection prompt to clipboard
        await this.copyConnectionPromptToClipboard();

        console.log(`\n📝 Paste the command above into your da_code agent to enable remote clipboard access.`);
        console.log(`⏹️  Press Ctrl+C to stop the server\n`);

        this.app.listen(this.port, '0.0.0.0', () => {
            console.log(`🚀 ClipJS server running on http://0.0.0.0:${this.port}`);
        });

        // Handle graceful shutdown
        process.on('SIGINT', () => {
            console.log('\n👋 ClipJS stopped');
            process.exit(0);
        });
    }
}