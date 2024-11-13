import * as path from "path";

const _DEV_STORAGE_BASE = path.join(path.dirname(process.cwd()), "tmp_storage");

const extractFileNameFromUploadUrl = (uploadUrl: string): string => {
  // Get 36 characters that end with ?
  const endIndex = uploadUrl.indexOf("?");
  return uploadUrl.substring(endIndex - 36, endIndex);
};

// export const mockUploadFile = async (
//   folder: string,
//   filePath: string,
//   uploadUrl: string,
// ): Promise<void> => {
//   const destFileName = extractFileNameFromUploadUrl(uploadUrl);
//   console.log(destFileName);
//   const fileLocation = path.join(_DEV_STORAGE_BASE, folder);
//   console.log(`Copying ${filePath} to ${fileLocation}`);
//
//   // Create localhost storage folder if needed
//   if (!fs.existsSync(fileLocation)) {
//     console.log(`Creating tmp storage ${fileLocation}`);
//     fs.mkdirSync(fileLocation, { recursive: true });
//   }
//
//   const destinationPath = path.join(fileLocation, destFileName);
//   fs.copyFileSync(filePath, destinationPath);
// };
