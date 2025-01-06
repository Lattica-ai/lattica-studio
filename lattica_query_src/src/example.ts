import { LatticaQueryClient, LatticaQueryClientLocalDev } from "./index";
import { exampleRunImage } from "./dev_utils";

const RUN_LOCAL = true;
const jwt =
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiZGZlYzA1MzktMGZlYy00NDAyLWFkMmItMzdhODdmOTE4OTVjIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzM2MDgyNjYzLCJleHAiOjE3Mzg2NzQ2NjMsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.Lg2i89-K0obWAlEIjJ95a7TU_JKL5aRwwyEQvwbao54Mf_RvEAkTAba0OpoeoxjCU3Tp-GBb13D5AkSB3NfpRMkJwocWhjTcmKLo0mk2abW62Li3kTCx3ie5PTAvJPCtfCl121QMlU20RD6tkHWaToIr6mb3zzzqzaOhN2CVDHCDcssw2pRwnoao8c3TQAAmzP4kH8igR-tpJ8nbI86RjWTZmkXdQOAqDfwAfwR0YfJ-mchWFm7AXfR0fbwFHkVV8Ha8urEeXDANicMX6A3wbImQCGcBRmyeyRfdQFnCj4lC015pUG6xLoQQu8eHxlPqGyZvIJ2kcKMAsk8eKrLCfQ";

let client;
if (RUN_LOCAL) {
  client = new LatticaQueryClientLocalDev(jwt);
  console.log("DEV client created");
} else {
  client = new LatticaQueryClient(
    jwt,
    "https://api-alb-509680522.eu-central-1.elb.amazonaws.com"  // old-dev env
  );
  console.log("Client created");
}

// number 9
let pt = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 97, 222, 253, 253, 128, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 252, 252, 252, 252, 253, 194, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 32, 222, 252, 252, 252, 252, 253, 252, 202, 70, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 116, 253, 252, 227, 66, 45, 169, 252, 252, 252, 232, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 116, 253, 252, 79, 0, 0, 13, 211, 252, 252, 252, 106, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 231, 255, 253, 69, 0, 0, 3, 170, 253, 253, 253, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 209, 253, 252, 90, 0, 0, 118, 252, 252, 252, 252, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 116, 253, 252, 234, 184, 184, 253, 252, 252, 252, 252, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 21, 211, 252, 252, 252, 252, 253, 252, 252, 252, 252, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 158, 252, 252, 252, 253, 178, 168, 252, 252, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 116, 0, 0, 81, 253, 253, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 252, 252, 116, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 252, 252, 32, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 240, 252, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151, 252, 252, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 170, 253, 253, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 252, 252, 252, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 252, 252, 252, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 252, 252, 218, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 158, 221, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
pt = pt.map((row) => row.map((value) => value / 255));

// number 4
const imageData =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA0ElEQVR4nGNgGAqA9+T/UpySs/7+xSUp1PDz73cnNMEYAQgd9/fvVy80uchfsyFqvv79ux1NLvTu3yviQFr4DNBCXjTJ13//1oJo079/H/KjSqm2//r7UBrIMLz192MYpuMfGoIY1n//5qE7/8ffv8+PzHByctrw928IuuStvwiwFV1SseHk3fNPTp78+PfvQRt0SSCQ49JgYDjy94sDFjkw4DmLaSgcePz9645Tcsbfm9K45PRf/N2BU6P/37+OOCV9/9ySwymp+aQSpxwVAQDGtGAPKDnpyQAAAABJRU5ErkJggg==";

// can run on base64 image data (imageData) or tensor (pt)
exampleRunImage(client, imageData).then(() => {});