import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
await build({entryPoints:['src/Dashboard.jsx'],bundle:true,format:'esm',platform:'browser',target:'es2020',outfile:'dashboard.js',minify:true,legalComments:'eof',define:{'process.env.NODE_ENV':'"production"'}});
const require=createRequire(import.meta.url);
const domRequire=createRequire(require.resolve('react-dom/package.json'));
const licenses=await Promise.all(['react','react-dom','scheduler'].map(async name=>`${name}\n${await readFile(join(dirname((name==='scheduler'?domRequire:require).resolve(`${name}/package.json`)),'LICENSE'),'utf8')}`));
await writeFile('THIRD_PARTY_LICENSES.txt',licenses.join('\n\n'));
