using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.AzureAISearch;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Plugins.Memory;

var text1 = File.ReadAllText(@"Data/cosine.txt");
var text2 = File.ReadAllText(@"Data/dot.txt");

#pragma warning disable SKEXP0001, SKEXP0020, SKEXP0010, SKEXP0050

// Azure OpenAI keys, models, endpoints
var gptDeployment = "gpt35turbo";
var gptModel = "gpt-35-turbo";
var aoaiEndpoint = "";
var aoaiKey = "";
var embeddingsdeployment = "embeddings";
var embeddingsmodel = "text-embedding-ada-002";

// Azure AI Search keys, endpoints, index
var searchEndpoint = "";
var searchKey = "";
var searchIndex = "text-index";

// Build the kernel
var builder = Kernel.CreateBuilder();
builder.Services.AddAzureOpenAIChatCompletion(gptDeployment, aoaiEndpoint, aoaiKey, gptModel);
var kernel = builder.Build();

// Build the memory: Azure OpenAI Text Embeddings and Azure AI Search
var memoryBuilder = new MemoryBuilder();
memoryBuilder.WithAzureOpenAITextEmbeddingGeneration(embeddingsdeployment, aoaiEndpoint, aoaiKey);

var aoaiSearchStore = new AzureAISearchMemoryStore(searchEndpoint, searchKey);
memoryBuilder.WithMemoryStore(aoaiSearchStore);

var memory = memoryBuilder.Build();

// Save information to memory: Generate embeddings and store them in Azure AI Search
await memory.SaveInformationAsync(searchIndex, id: "doc-1", text: text1);
await memory.SaveInformationAsync(searchIndex, id: "doc-2", text: text2);

// Build a Prompt Function
kernel.ImportPluginFromObject(new TextMemoryPlugin(memory));

var prompt = @"
AIBot can have a conversation and answer any question about AI.
It says 'I don't know' if it doesn't have an answer.

Information from previous conversations:
- {{$fact1}} {{recall $fact1}}
- {{$fact2}} {{recall $fact2}}

Chat:
{{$history}}
User: {{$userInput}}
AIBot: ";

var settings = new OpenAIPromptExecutionSettings { MaxTokens = 500, Temperature = 0.8 };
var chatFunction = kernel.CreateFunctionFromPrompt(prompt, settings);

// Chat with the AI Bot: Define arguments, set history
var history = "";

var arguments = new KernelArguments
{
    { "fact1", "cosine similarity" },
    { "fact2", "dot product" },
    { TextMemoryPlugin.CollectionParam, searchIndex },
    { TextMemoryPlugin.LimitParam, "2" },
    { TextMemoryPlugin.RelevanceParam, "0.7" },
    { history, history }
};

// Chat with the AI Bot: Chat function
async Task Chat(string input)
{
    arguments["userInput"] = input;
    var answer = await chatFunction.InvokeAsync(kernel, arguments);
    var result = $"Response: {answer}\n";

    Console.WriteLine($"Question: {input}");
    Console.WriteLine($"AIBot: {result}\n");

    history += result;
    arguments["history"] = history;
}

// Chat with the AI Bot: Chat chat chat!
await Chat("What are two applications of dot product?");
await Chat("What is cosine similarity, in less than 50 words?");
await Chat("Is Collaborative filtering an application of cosine similarity?");
