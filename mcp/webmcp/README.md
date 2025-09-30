# 🌐 webmcp - Browser Automation MCP Server

[![Node.js](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org)
[![Playwright](https://img.shields.io/badge/playwright-automation-blue.svg)](https://playwright.dev)
[![MCP](https://img.shields.io/badge/MCP-protocol-orange.svg)](https://github.com/anthropics/mcp)

> **Browser automation MCP server using official Playwright MCP with da_code integration**

## 🚀 The Magic

**webmcp** follows our **revolutionary local MCP pattern**:

1. **Start locally**: `webmcp` (starts Playwright MCP server)
2. **Auto-copies connection**: JSON config copied to clipboard
3. **Remote connection**: Paste in da_code → instant browser automation
4. **Full power**: Remote AI agent controls browser via Playwright tools

## ⚡ Key Features

### 🎭 **Official Playwright MCP Integration**
- **Wraps** `@modelcontextprotocol/server-playwright`
- **All tools** from official Playwright MCP server
- **Real browser** automation (not headless by default)
- **Cross-platform** browser support

### 🌐 **Full Browser Automation**
- **Navigate** to URLs with wait conditions
- **Click** elements using selectors
- **Fill forms** and interact with inputs
- **Take screenshots** (full page or viewport)
- **Execute JavaScript** with return values
- **Wait** for elements or conditions
- **Multi-tab** management
- **Network monitoring**
- **Console access**

### 🔄 **da_code Integration**
- **Automatic clipboard copy** of connection JSON
- **Session-scoped** connections (clean disconnection)
- **Real-time** browser control from remote agents
- **Error handling** with detailed feedback

## 🎯 Installation & Usage

### **Local Setup** (Your machine with browser)
```bash
# Install webmcp
cd /mnt/blk/lostboy/mcp/webmcp
npm install

# Install globally for webmcp command
npm install -g .

# Start server
webmcp

# Output:
# ✅ Connection command copied to clipboard!
# 📋 Browser MCP Connection Config:
# {
#   "name": "webmcp",
#   "url": "http://192.168.1.77:8005",
#   "description": "Browser automation from YourMachine",
#   "tools": ["browser_navigate", "browser_click", ...]
# }
```

### **Remote Connection** (da_code instance)
```bash
# In your da_code session (Linux/cloud)
> add_mcp {"name": "webmcp", "url": "http://192.168.1.77:8005", ...}
✅ MCP server 'webmcp' added successfully

# Instantly available tools:
> browser_navigate {"url": "https://github.com"}
🤖 Navigated to GitHub using your local browser!
```

## 🛠️ Available Tools

All tools from `@modelcontextprotocol/server-playwright`:

### **Navigation & Control**
- `browser_navigate` - Navigate to URLs
- `browser_click` - Click elements
- `browser_type` - Type text into elements
- `browser_fill_form` - Fill multiple form fields
- `browser_press_key` - Press keyboard keys
- `browser_hover` - Hover over elements
- `browser_drag` - Drag and drop elements

### **Content & Screenshots**
- `browser_screenshot` - Take screenshots
- `browser_take_screenshot` - Enhanced screenshot tool
- `browser_snapshot` - Accessibility snapshot
- `browser_evaluate` - Execute JavaScript

### **Waiting & Timing**
- `browser_wait_for` - Wait for conditions

### **Advanced Features**
- `browser_tabs` - Tab management
- `browser_resize` - Resize browser window
- `browser_console_messages` - Access console
- `browser_network_requests` - Monitor network
- `browser_handle_dialog` - Handle dialogs
- `browser_file_upload` - Upload files
- `browser_navigate_back` - Navigate back
- `browser_close` - Close browser
- `browser_install` - Install browser

## 🎪 Usage Examples

### **Quick Navigation**
```bash
> browser_navigate {"url": "https://example.com"}
🤖 ✅ Navigated to https://example.com

> browser_click {"element": "Sign In button", "ref": "#signin"}
🤖 ✅ Clicked Sign In button
```

### **Form Interaction**
```bash
> browser_fill_form {
    "fields": [
      {"name": "username", "type": "textbox", "ref": "#user", "value": "myuser"},
      {"name": "password", "type": "textbox", "ref": "#pass", "value": "secret"}
    ]
  }
🤖 ✅ Filled 2 form fields successfully
```

### **Screenshots & Content**
```bash
> browser_take_screenshot {"fullPage": true, "type": "png"}
🤖 📸 Screenshot taken and saved

> browser_evaluate {"function": "() => document.title"}
🤖 📄 Page title: "Example Domain"
```

## ⚙️ Configuration Options

### **Browser Selection**
```bash
# Default port
webmcp

# Custom port
webmcp --port 9000

# Headless mode
webmcp --headless
```

## 🏗️ Architecture

### **MCP Integration Flow**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Local Browser  │◄──►│     webmcp      │◄──►│ Remote da_code  │
│                 │    │                 │    │                 │
│ 🎭 Playwright   │    │ 🔌 MCP Wrapper  │    │ 🤖 AI Agent     │
│ 🌐 Real Browser │    │ 📋 Auto-copy    │    │ 🧠 Commands     │
│ 🔑 Your Session │    │ 🚀 Node.js      │    │ 🔗 MCP Client   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Technology Stack**
- **Core**: `@modelcontextprotocol/server-playwright`
- **Runtime**: Node.js 18+
- **Browser**: Playwright (Chromium/Firefox/WebKit)
- **Clipboard**: `clipboardy` for auto-copy
- **CLI**: `commander` for argument parsing

## 🚨 Security Considerations

### **Safe Defaults**
- **Non-headless by default**: See what the browser is doing
- **Local binding**: Server runs on local network interface
- **Session isolation**: Each connection is independent
- **Clean shutdown**: Proper resource cleanup

### **Network Security**
- **Local network only**: No internet exposure by default
- **Known IP binding**: Uses actual local IP for connection
- **Port configuration**: Configurable port for flexibility

## 🎯 Perfect for da_code

**webmcp** integrates seamlessly with the da_code ecosystem:

1. **Official MCP tools**: Uses standard Playwright MCP server
2. **Auto-discovery**: Clipboard copy for instant connection
3. **Session management**: Clean connection lifecycle
4. **Rich responses**: Detailed success/error feedback
5. **Cross-platform**: Works on Windows, macOS, Linux

## 🔧 Development

### **Local Development**
```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Test connection
node bin/webmcp.js --port 8005
```

### **Dependencies**
- `@modelcontextprotocol/server-playwright` - Official Playwright MCP
- `commander` - CLI argument parsing
- `clipboardy` - Cross-platform clipboard access

## 🔮 Future Enhancements

- **Browser profile support**: Custom user data directories
- **Extension integration**: Browser extension communication
- **Mobile browsers**: iOS Safari and Android Chrome
- **Performance metrics**: Page load timing and analysis
- **Video recording**: Capture browser interactions

---

**webmcp: Official Playwright MCP with da_code magic.**

**Simple setup. Powerful automation. Unlimited possibilities.**