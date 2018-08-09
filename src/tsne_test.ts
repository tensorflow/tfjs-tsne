/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';
import * as tf_tsne from './tsne';

/**
 * Returns a simple dataset for testing
 * Defaults to numPoints = 300, numDimensions = 10
 */
function generateData(numPoints = 300, numDimensions = 10) {
  const data = tf.tidy(() => {
    return tf.linspace(0, 1, numPoints * numDimensions)
        .reshape([ numPoints, numDimensions ])
        .add(tf.randomUniform([ numPoints, numDimensions ]));
  });
  return data;
}

describe('TSNE class', () => {
  it('throws an error if the perplexity is too high', () => {
    const data = generateData();
    expect(() => {
      tf_tsne.tsne(data, {
        perplexity : 100,
        verbose : false,
        knnMode : 'auto',
      });
    }).toThrow();

    data.dispose();
  });
});

describe('TSNE class', () => {
  it('throws an error if the perplexity is too high on this system ', () => {
    const data = generateData();
    const maximumPerplexity = tf_tsne.maximumPerplexity();
    expect(() => {
      tf_tsne.tsne(data, {
        perplexity : maximumPerplexity + 1,
        verbose : false,
        knnMode : 'auto',
      });
    }).toThrow();

    data.dispose();
  });
});

describe('TSNE class', () => {
  it('does not throw an error if the perplexity is set to the maximum value',
     () => {
       const data = generateData();
       const maximumPerplexity = tf_tsne.maximumPerplexity();
       expect(() => {
         tf_tsne.tsne(data, {
           perplexity : maximumPerplexity,
           verbose : false,
           knnMode : 'auto',
         });
       }).not.toThrow();

       data.dispose();
     });
});

describe('TSNE class', () => {
  it('iterateKnn and iterate also work when the number of ' +
    'dimensions is larger than the number of points',
    async () => {
      const data = generateData(100, 20000);
      const testOpt = tf_tsne.tsne(data, {
          perplexity : 15,
          verbose : false,
          knnMode : 'auto',
      });

      try {
        await testOpt.iterateKnn(10);
      } catch(e) {
        fail('iterateKnn threw exception: ${e}');
      }

      try {
        await testOpt.iterate(10);
      } catch(e) {
        fail('iterate threw exception: ${e}');
      }

      const coords = await testOpt.coordinates();
      expect(coords.shape[0]).toBe(100);
      expect(coords.shape[1]).toBe(2);
      data.dispose();
      return;
    });
});