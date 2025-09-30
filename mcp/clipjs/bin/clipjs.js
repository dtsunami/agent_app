#!/usr/bin/env node

import { Command } from 'commander';
import { ClipjsServer } from '../src/server.js';

const program = new Command();

program
  .name('clipjs')
  .description('ClipJS - Node.js MCP Server for Windows clipboard access')
  .version('1.0.0')
  .option('-p, --port <number>', 'port to run server on', '3000')
  .parse();

const options = program.opts();
const port = parseInt(options.port);

const server = new ClipjsServer(port);
server.start().catch(console.error);