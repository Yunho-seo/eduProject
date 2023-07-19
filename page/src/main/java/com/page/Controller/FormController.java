package com.page.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.ui.Model;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import org.springframework.context.annotation.Configuration;

import org.springframework.context.annotation.Bean;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

@Controller
public class FormController {

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam(value="image", required=false) MultipartFile multipartFile)
            throws IOException {

        String flaskAppUrl = "http://localhost:5000/upload";

        // MultipartFile을 File로 변환
        File convertedFile = convertMultipartFileToFile(multipartFile);

        // 플라스크 서버로 이미지 전송
        ResponseEntity<byte[]> responseEntity = new RestTemplate().postForEntity(
                flaskAppUrl + "/upload", convertedFile, byte[].class);

        // 플라스크 서버로부터의 응답을 파일로 저장
        File receivedImageFile = saveReceivedImage(responseEntity.getBody());

        // 모델에 전달할 파일 경로 설정
        // model.addAttribute("imagePath", receivedImageFile.getAbsolutePath());

        return "uploadsuccess";
    }

    private File convertMultipartFileToFile(MultipartFile file) throws IOException {
        File convertedFile = new File(file.getOriginalFilename());
        try (FileOutputStream fos = new FileOutputStream(convertedFile)) {
            fos.write(file.getBytes());
        }
        return convertedFile;
    }

    private File saveReceivedImage(byte[] imageBytes) throws IOException {
        File receivedImageFile = new File("received_image.jpg");
        try (FileOutputStream fos = new FileOutputStream(receivedImageFile)) {
            fos.write(imageBytes);
        }
        return receivedImageFile;
    }
}