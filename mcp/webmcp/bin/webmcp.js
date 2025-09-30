#!/usr/bin/env node

import { Command } from 'commander';
import { WebMCPServer } from '../src/server.js';

const program = new Command();

program
  .name('webmcp')
  .description('WebMCP - Browser automation MCP server using Playwright')
  .version('1.0.0')
  .option('-p, --port <number>', 'port to run server on', '8005')
  .option('--headless', 'run browser in headless mode', false)
  .parse();

const options = program.opts();
const port = options.port ? parseInt(options.port) : undefined; // Only parse if provided
const headless = options.headless;

const server = new WebMCPServer(port, headless);
server.start().catch(console.error);