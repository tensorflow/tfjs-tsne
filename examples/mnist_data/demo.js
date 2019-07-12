let counter = 0;
let dataCount = 0;
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function step(timestamp) {
  counter++;
  console.log(`Animate count: ${counter}`);
  await sleep(200);
  if (counter < 30) {
    requestAnimationFrame(step);
  }
}


// async generator
async function* genAsyncData() {
  while (dataCount < 6) {
    await sleep(1000)
    yield dataCount;
    dataCount++;
  }
}

(async function main() {
  // for-await syntax
  const gAD = genAsyncData();
  console.log('start RAF');
  window.requestAnimationFrame(step);
  console.log('started RAF');
  for await (const genCount of genAsyncData()) {
    console.log(`Generator count: ${genCount}`);
  }
}());