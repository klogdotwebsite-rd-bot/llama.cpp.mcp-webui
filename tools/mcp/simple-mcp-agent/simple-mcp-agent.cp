//
// A complete and working MCP agent that provides a safe 'shell_command' tool.
// 1. Uses a local Llama model for inference.
// 2. Implements a full agent loop, feeding tool results back to the model.
// 3. Maintains the MCP server/client architecture for tool execution.
//
// Usage:
// Required:
//   -m <path>       Path to the GGUF model file
// Optional:
//   --port <n>      Server port (default: 8889)
//   --confirm       Require user confirmation before executing shell commands
//

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

using json = nlohmann::json;

//================================================================================
// TOOL IMPLEMENTATION
//================================================================================

// A helper to clean up potential markdown/chat markers from the LLM's output
std::string clean_llm_response(const std::string& response) {
    std::string result = response;
    std::vector<std::string> markers_to_remove = {
        "```json", "```", "<|im_start|>", "<|im_end|>",
        "<|assistant|>", "<|user|>", "assistant\n", "user\n"
    };

    for (const auto& marker : markers_to_remove) {
        size_t pos;
        while ((pos = result.find(marker)) != std::string::npos) {
            result.erase(pos, marker.length());
        }
    }
    result.erase(0, result.find_first_not_of(" \n\r\t"));
    result.erase(result.find_last_not_of(" \n\r\t") + 1);
    return result;
}

// A safe shell command execution tool handler
class ShellCommandHandler {
public:
    static mcp::json handle(const mcp::json& params, const std::string& /* session_id */) {
        if (!params.contains("command")) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'command' parameter");
        }

        std::string cmd = clean_llm_response(params["command"]);
        if (cmd.empty()) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Empty command provided");
        }

        if (!is_command_safe(cmd)) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Command not allowed for security reasons. Only basic inspection commands are permitted.");
        }

        try {
            std::string result = execute_command(cmd);
            return {
                { "type", "text" },
                { "text", result }
            };
        } catch (const std::exception& e) {
            throw mcp::mcp_exception(mcp::error_code::internal_error, e.what());
        }
    }

private:
    static bool is_command_safe(const std::string& cmd) {
        const std::vector<std::string> allowed_prefixes = { "ls", "pwd", "echo", "cat", "date", "whoami", "uname" };
        const std::vector<std::string> blocked_tokens = { "rm", "sudo", "su", ">", ">>", "|", "mv", "cp", "chmod", "chown", "&", ";" };

        for (const auto& token : blocked_tokens) {
            if (cmd.find(token) != std::string::npos) return false;
        }
        for (const auto& prefix : allowed_prefixes) {
            if (cmd.rfind(prefix, 0) == 0) return true;
        }
        return false;
    }

    static std::string execute_command(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("Failed to execute command with popen.");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }
};

//================================================================================
// AGENT CONFIGURATION AND SETUP
//================================================================================

struct Config {
    std::string model_path;
    int port = 8889;
    bool confirm_commands = false;
    int n_gpu_layers = 99;
    int n_ctx = 2048;
    int n_batch = 512;
    std::string system_prompt = "You are a helpful assistant with access to a shell. When the user asks for something that requires a command, you must first think about which command to use. Then, you must call the 'shell_command' tool with that command. After you receive the result, you must formulate a final answer to the user based on the tool's output.";
};

static Config parse_config(int argc, char* argv[]) {
    Config config;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--confirm") == 0) {
            config.confirm_commands = true;
        }
    }
    return config;
}

static bool readline_utf8(std::string & line) {
#if defined(_WIN32)
    std::wstring wline;
    if (!std::getline(std::wcin, wline)) { line.clear(); return false; }
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), NULL, 0, NULL, NULL);
    line.resize(size_needed);
    WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), &line[0], size_needed, NULL, NULL);
#else
    if (!std::getline(std::cin, line)) { line.clear(); return false; }
#endif
    return true;
}

//================================================================================
// MAIN AGENT LOGIC
//================================================================================

int main(int argc, char ** argv) {
    Config config = parse_config(argc, argv);
    if (config.model_path.empty()) {
        fprintf(stderr, "Error: Model path (-m) is required.\n");
        return 1;
    }

    // --- LLAMA MODEL INITIALIZATION ---
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model) { fprintf(stderr, "Error: Failed to load model from %s\n", config.model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "Error: Failed to create llama_context\n"); return 1; }

    // --- MCP SERVER AND TOOL SETUP ---
    mcp::server server("localhost", config.port);
    server.set_server_info("ShellAgentServer", "0.1.0");
    server.set_capabilities({{"tools", json::object()}});
    mcp::tool shell_tool = mcp::tool_builder("shell_command")
        .with_description("Execute a basic, safe shell command.")
        .with_string_param("command", "The shell command to execute (e.g., 'ls -l').")
        .build();
    server.register_tool(shell_tool, ShellCommandHandler::handle);
    server.start(false);

    // --- MCP CLIENT SETUP ---
    mcp::sse_client client("localhost", config.port);
    client.set_timeout(10);
    if (!client.initialize("ShellAgentClient", "0.1.0")) { fprintf(stderr, "Error: Failed to initialize MCP client.\n"); return 1; }

    // --- AGENT INITIALIZATION ---
    std::vector<common_chat_tool> llm_tools = {{shell_tool.name, shell_tool.description, shell_tool.parameters_schema.dump()}};
    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, "");
    std::vector<common_chat_msg> messages = {{"system", config.system_prompt, {}, {}, "", "", ""}};

    // --- MAIN INTERACTIVE LOOP ---
    printf("Shell MCP Agent is running. Press Ctrl+C to exit.\n");
    std::string line;
    while (true) {
        printf("\n> ");
        if (!readline_utf8(line) || line == "exit") break;
        if (line.empty()) continue;

        messages.push_back({"user", line, {}, {}, "", "", ""});

        while (true) {
            common_chat_templates_inputs inputs = {messages, llm_tools, COMMON_CHAT_TOOL_CHOICE_AUTO, true, true};
            auto chat_params = common_chat_templates_apply(chat_templates.get(), inputs);

            const int n_prompt = -llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), NULL, 0, true, true);
            std::vector<llama_token> prompt_tokens(n_prompt);
            llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

            std::string response_text;
            printf("Assistant: ");
            for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + 256; ) {
                if (llama_decode(ctx, batch)) { fprintf(stderr, "Failed to eval\n"); return 1; }
                n_pos += batch.n_tokens;
                llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, new_token_id)) break;
                char buf[128];
                int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0) { fprintf(stderr, "Failed to convert token\n"); return 1; }
                std::string s(buf, n);
                response_text += s;
                printf("%s", s.c_str());
                fflush(stdout);
                batch = llama_batch_get_one(&new_token_id, 1);
            }
            printf("\n");
            llama_sampler_free(smpl);

            common_chat_syntax syntax = {chat_params.format, true};
            common_chat_msg parsed_response = common_chat_parse(response_text, false, syntax);
            messages.push_back(parsed_response);

            if (parsed_response.tool_calls.empty()) break;

            for (const auto& tool_call : parsed_response.tool_calls) {
                common_chat_msg tool_response_msg = {"tool", "", {}, {}, "", "", tool_call.id};
                try {
                    mcp::json args = json::parse(clean_llm_response(tool_call.arguments));
                    if (config.confirm_commands) {
                        printf("    Execute command '%s'? (y/N): ", args["command"].get<std::string>().c_str());
                        std::string confirm_line;
                        std::getline(std::cin, confirm_line);
                        if (confirm_line != "y" && confirm_line != "Y") {
                            printf("    Execution cancelled by user.\n");
                            tool_response_msg.content = "{\"type\":\"text\", \"text\":\"Command execution cancelled by user.\"}";
                            messages.push_back(tool_response_msg);
                            continue;
                        }
                    }
                    mcp::json result = client.call_tool(tool_call.name, args);
                    tool_response_msg.content = result.value("content", json::array()).dump();
                } catch (const std::exception& e) {
                    tool_response_msg.content = std::string("{\"type\":\"text\", \"text\":\"Error: ") + e.what() + "\"}";
                }
                messages.push_back(tool_response_msg);
            }
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}