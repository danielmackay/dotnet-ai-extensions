using Azure;
using Azure.AI.Inference;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var endpoint = new Uri("https://models.inference.ai.azure.com");
var credential = new AzureKeyCredential(config["GH_PAT"]!);

IChatClient client =
    new ChatCompletionsClient(
            endpoint: endpoint,
            credential)
        .AsChatClient("Phi-3.5-MoE-instruct");

var response = await client.CompleteAsync("What is AI?");

Console.WriteLine(response.Message);

IEmbeddingGenerator<string,Embedding<float>> generator =
    new AzureOpenAIClient(
            endpoint,
            credential)
        .AsEmbeddingGenerator(modelId: "text-embedding-3-small");

var embeddings = await generator.GenerateAsync(["What is AI?"]);

Console.WriteLine(string.Join(", ", embeddings[0].Vector.ToArray()));
