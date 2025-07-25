#include "llama.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "json.hpp"
#include "mcp_server.h"
#include "mcp_sse_client.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <array>
#include <map>
#include <sstream>

using json = nlohmann::json;

// Configuration for the client, holding server details
struct Config {
    struct ServerConfig {
        std::string name;
        std::string host;
        int port;
        std::string type;
    };
    std::vector<ServerConfig> servers;
    bool show_instructions = true;
};

// Represents a single connected MCP server and its capabilities
struct MCPServerConnection {
    std::string name;
    std::string type;
    std::unique_ptr<mcp::sse_client> client;
    std::vector<mcp::tool> tools;
};

// Global state for the application
std::vector<MCPServerConnection> connected_servers;
std::map<std::string, std::string> tool_to_server_map; // Maps tool name to server name

// Parses command-line arguments to configure servers
static Config parse_config(int argc, char* argv[]) {
    Config config;
    // Add a default server for convenience
    config.servers.push_back({"default-agent", "localhost", 8889, "llama-agent"});

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--add-server") == 0 && i + 4 < argc) {
            config.servers.push_back({
                argv[i+1], // name
                argv[i+2], // host
                std::stoi(argv[i+3]), // port
                argv[i+4]  // type
            });
            i += 4;
        } else if (strcmp(argv[i], "--hide-instructions") == 0) {
            config.show_instructions = false;
        }
    }
    return config;
}

// Establishes a connection to a configured MCP server
bool connect_to_server(const Config::ServerConfig& server_config, MCPServerConnection& server) {
    try {
        server.name = server_config.name;
        server.type = server_config.type;
        server.client = std::make_unique<mcp::sse_client>(server_config.host, server_config.port);
        server.client->set_timeout(5); // 5 second timeout

        if (!server.client->initialize("llama-mcp-client", "0.1.0")) {
            fprintf(stderr, "Error: Failed to initialize connection to '%s' at %s:%d\n",
                    server.name.c_str(), server_config.host.c_str(), server_config.port);
            return false;
        }

        server.tools = server.client->get_tools();
        for (const auto& tool : server.tools) {
            tool_to_server_map[tool.name] = server.name;
        }

        printf("Successfully connected to '%s' (%zu tools found)\n", server.name.c_str(), server.tools.size());
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: Exception while connecting to '%s': %s\n", server.name.c_str(), e.what());
        return false;
    }
}

// Displays all available tools grouped by server
void display_tools() {
    printf("\n--- Available Tools ---\n");
    if (tool_to_server_map.empty()) {
        printf("No tools found on any connected servers.\n");
        return;
    }
    for (const auto& server : connected_servers) {
        if (!server.tools.empty()) {
            printf("\nFrom server '%s' (%s):\n", server.name.c_str(), server.type.c_str());
            for (const auto& tool : server.tools) {
                printf("  - %s: %s\n", tool.name.c_str(), tool.description.c_str());
            }
        }
    }
    printf("\n");
}

// Finds the server connection responsible for a given tool
MCPServerConnection* find_server_for_tool(const std::string& tool_name) {
    auto it = tool_to_server_map.find(tool_name);
    if (it == tool_to_server_map.end()) {
        return nullptr;
    }
    for (auto& server : connected_servers) {
        if (server.name == it->second) {
            return &server;
        }
    }
    return nullptr;
}

// Executes a tool call on the appropriate server
void execute_tool(const std::string& tool_name, const json& args) {
    auto* server = find_server_for_tool(tool_name);
    if (!server) {
        fprintf(stderr, "Error: Tool '%s' not found on any connected server.\n", tool_name.c_str());
        return;
    }

    printf("Executing tool '%s' on server '%s'...\n", tool_name.c_str(), server->name.c_str());
    try {
        json result = server->client->call_tool(tool_name, args);
        printf("\nResult:\n%s\n", result.dump(2).c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: Exception during tool execution: %s\n", e.what());
    }
}

// The main interactive command loop
void run_interactive_mode(const Config& config) {
    if (config.show_instructions) {
        printf("\n--- MCP Client Interactive Mode ---\n");
        printf("Commands:\n");
        printf("  - tools                            List all available tools.\n");
        printf("  - tool <name> <json_args>          Execute a tool (e.g., tool calculator '{\"expression\":\"2+2\"}').\n");
        printf("  - servers                          List all connected servers.\n");
        printf("  - help                             Show this help message.\n");
        printf("  - exit                             Quit the client.\n");
    }

    std::string line;
    while (true) {
        printf("\nmcp> ");
        std::getline(std::cin, line);
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "exit" || command == "quit") {
            break;
        } else if (command == "tools") {
            display_tools();
        } else if (command == "servers") {
            printf("\n--- Connected Servers ---\n");
            for (const auto& server : connected_servers) {
                printf("- %s (%s) at %s:%d\n", server.name.c_str(), server.type.c_str(), server.client->get_host().c_str(), server.client->get_port());
            }
        } else if (command == "tool") {
            std::string tool_name;
            ss >> tool_name;
            if (tool_name.empty()) {
                fprintf(stderr, "Error: Tool name is required. Usage: tool <name> <json_args>\n");
                continue;
            }

            std::string args_str;
            std::getline(ss, args_str);
            // Trim leading whitespace from args
            args_str.erase(0, args_str.find_first_not_of(" \t\n\r"));

            json args;
            if (args_str.empty()) {
                args = json::object(); // No args provided
            } else {
                try {
                    args = json::parse(args_str);
                } catch (const std::exception& e) {
                    fprintf(stderr, "Error: Invalid JSON arguments: %s\n", e.what());
                    continue;
                }
            }
            execute_tool(tool_name, args);
        } else if (command == "help") {
             run_interactive_mode({true}); // Show instructions again
        }else {
            fprintf(stderr, "Unknown command: '%s'. Type 'help' for a list of commands.\n", command.c_str());
        }
    }
}

int main(int argc, char** argv) {
    Config config = parse_config(argc, argv);

    printf("Starting MCP client...\n");
    for (const auto& server_config : config.servers) {
        MCPServerConnection server;
        if (connect_to_server(server_config, server)) {
            connected_servers.push_back(std::move(server));
        }
    }

    if (connected_servers.empty()) {
        fprintf(stderr, "\nFatal: No servers could be connected. Please check your server configurations.\n");
        return 1;
    }

    run_interactive_mode(config);

    printf("Exiting MCP client.\n");
    return 0;
}