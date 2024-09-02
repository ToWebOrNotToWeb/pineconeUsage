import { Pinecone } from '@pinecone-database/pinecone';
import dotenv from 'dotenv';
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";

dotenv.config();

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});

await pc.createIndex({
    name: 'market-place',
    dimension: 1536, 
    metric: 'cosine',
    spec: {
      serverless: {
        cloud: 'aws',
        region: 'us-east-1'
      }
    }
});

const openai = new OpenAI();
const documentsFolder = "./dataLight";
const chunkSize = 3000;  

async function extractTextFromPDF(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const pdfData = await pdfParse(dataBuffer);
  return pdfData.text;
}

function splitText(text, chunkSize) {
  const chunks = [];
  let start = 0;
  
  while (start < text.length) {
    let end = start + chunkSize;

    // Avoid breaking in the middle of a word.
    if (end < text.length) {
      end = text.lastIndexOf(` `, end);
    }

    chunks.push(text.slice(start, end).trim());
    start = end;
  }

  return chunks;
}

async function main() {
  const files = fs.readdirSync(documentsFolder).filter(file => file.endsWith(".pdf"));
  const results = [];

  for (const file of files) {
    const filePath = path.join(documentsFolder, file);
    const text = await extractTextFromPDF(filePath);

    const textChunks = splitText(text, chunkSize);
    
    for (let i = 0; i < textChunks.length; i++) {
      const embedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: textChunks[i],
        encoding_format: "float",
      });

      results.push({
        id: `vec${i + 1}`,
        values: embedding.data[0].embedding,
        metadata: {
          file,
          chunk: i + 1,
          text: textChunks[i],
        }
      });

      console.log(`Processed chunk ${i + 1} for ${file}`);
    }
  }

  console.log("All embeddings:", results);

  try {
    const index = pc.Index('market-place');
    await index.namespace("main").upsert(results);

  } catch (e) {
      console.error(e);
  }

}

main().catch(err => console.error(err));
