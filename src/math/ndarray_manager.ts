/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import * as util from '../util';
import {TypedArray} from '../util';
import {GPGPUContext} from './backends/webgl/gpgpu_context';
import {TextureType} from './backends/webgl/tex_util';
import {TextureManager} from './backends/webgl/texture_manager';
import * as webgl_util from './backends/webgl/webgl_util';

// These global variables need to be initialized to null so that closure knows
// not to seal them.
/** @hidden */
export let GPGPU: GPGPUContext = null;
/** @hidden */
export let TEXTURE_MANAGER: TextureManager = null;

export interface DataTypes {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}

export interface NDArrayData<T extends keyof DataTypes> {
  id: number;
  dtype: T;
  shape: number[];
}

export function initializeGPU(
    gpgpu: GPGPUContext, textureManager: TextureManager) {
  GPGPU = gpgpu;
  TEXTURE_MANAGER = textureManager;
}

function typedArrayToFloat32(
    a: TypedArray, dtype: keyof DataTypes): Float32Array {
  if (a instanceof Float32Array) {
    return a;
  } else {
    const res = new Float32Array(a.length);
    for (let i = 0; i < res.length; i++) {
      const val = a[i];
      res[i] = util.isValNaN(val, dtype) ? NaN : val;
    }
    return res;
  }
}

export interface StorageData {}

export interface NDArrayDataStorage {
  allocate(id: number, shape: number[], dtype: keyof DataTypes): StorageData;
  upload(
      id: number, shape: number[], dtype: keyof DataTypes,
      values: TypedArray): StorageData;
  download<T extends keyof DataTypes>(id: number): DataTypes[T];
  dispose(id: number): void;
}

export interface WebGLData {
  texture?: WebGLTexture;
  /** [rows, columns] shape of the texture. */
  texShape?: [number, number];
  type?: TextureType;
}

export class WebGLStorage implements NDArrayDataStorage {
  private db: {[id: number]: WebGLData};

  allocate(id: number, shape: number[], dtype: keyof DataTypes): WebGLData {
    this.throwIfDisposed();
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(GPGPU.gl, shape);
    const texture = TEXTURE_MANAGER.acquireTexture(texShape);
    const type = TextureType.DEFAULT;
    const webGLData = {texture, type, texShape};
    this.db[id] = webGLData;
    return webGLData;
  }

  upload(
      id: number, shape: number[], dtype: keyof DataTypes,
      values: TypedArray): WebGLData {
    const webGLData = this.allocate(id, shape, dtype);
    const {texture, texShape} = webGLData;
    // TODO(smilkov): Propagate the original typed array to gpgpu.
    GPGPU.uploadMatrixToTexture(
        texture, texShape[0], texShape[1], typedArrayToFloat32(values, dtype));
    return webGLData;
  }

  download<T extends 'float32'|'int32'|'bool'>(id: number): DataTypes[T] {
    this.throwIfDisposed();
    throw new Error('Method not implemented.');
  }

  dispose(id: number): void {
    this.throwIfDisposed();
    throw new Error('Method not implemented.');
  }

  throwIfDisposed(): void {
    throw new Error('Method not implemented.');
  }
}

export class NDArrayManager {
  constructor(private storage: NDArrayDataStorage) {}

  makeNDArrayData<T extends keyof DataTypes>(
      shape: number[], dtype: T, values?: TypedArray): NDArrayData<T> {
    const id = this.nextId++;
    const data: NDArrayData<T> = {shape, dtype, id};
    if (values != null) {
      this.storage.upload(id, shape, dtype, values);
    }
    this.track(data);
    return data;
  }

  private nextId = 1;
  private track(data: NDArrayData<keyof DataTypes>) {
    console.log('tracked', data);
  }
}
