import { Pinecone } from '@pinecone-database/pinecone';
import dotenv from 'dotenv';
import OpenAI from "openai";

dotenv.config();

const query = "Who is silvia carter ?";

const openai = new OpenAI();

const pc = new Pinecone({

    apiKey: process.env.PINECONE_API_KEY

});

const index = pc.index("twontw")

async function main() {

    const embedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: query,
        encoding_format: "float",
    });

    console.log(embedding);

    const queryResponse = await index.namespace('bookOne').query({
        vector: embedding.data[0].embedding,
        topK: 3,
        includeValues: false,
        includeMetadata: true
    });

    console.log(queryResponse);
    console.log(queryResponse.matches[0].metadata.text);
    console.log(queryResponse.matches[1].metadata.text);
    console.log(queryResponse.matches[2].metadata.text);

}

main();




