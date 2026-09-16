import { FlyBrain } from '../flybrain/flybrain.js';
import { makeAdditionReader } from '../hae_addition/addition_reader.mjs';

const ROOT = new URL('../', import.meta.url);
const R = await makeAdditionReader({ FlyBrain, base: new URL('flybrain/', ROOT) });

console.log('AdditionReader initialized successfully.');
console.log('Classes:', R.labels.length, R.labels);
console.log('Left PNs:', R.leftPNs.length, 'Right PNs:', R.rightPNs.length);

const groupSummary = R.groups.map((g, i) => `Sum ${i}: ${g.cells.length} MBONs, ${g.inputs} KC-inputs, ${g.synapses} synapses`);
console.log(groupSummary.join('\n'));
