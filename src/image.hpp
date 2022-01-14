  struct bitmap_file_header
   {
      unsigned short type;
      unsigned int   size;
      unsigned short reserved1;
      unsigned short reserved2;
      unsigned int   off_bits;

      unsigned int struct_size() const
      {
         return sizeof(type     ) +
                sizeof(size     ) +
                sizeof(reserved1) +
                sizeof(reserved2) +
                sizeof(off_bits ) ;
      }

      void clear()
      {
         std::memset(this, 0x00, sizeof(bitmap_file_header));
      }
   };

   struct bitmap_information_header
   {
      unsigned int   size;
      unsigned int   width;
      unsigned int   height;
      unsigned short planes;
      unsigned short bit_count;
      unsigned int   compression;
      unsigned int   size_image;
      unsigned int   x_pels_per_meter;
      unsigned int   y_pels_per_meter;
      unsigned int   clr_used;
      unsigned int   clr_important;

      unsigned int struct_size() const
      {
         return sizeof(size            ) +
                sizeof(width           ) +
                sizeof(height          ) +
                sizeof(planes          ) +
                sizeof(bit_count       ) +
                sizeof(compression     ) +
                sizeof(size_image      ) +
                sizeof(x_pels_per_meter) +
                sizeof(y_pels_per_meter) +
                sizeof(clr_used        ) +
                sizeof(clr_important   ) ;
      }

      void clear()
      {
         std::memset(this, 0x00, sizeof(bitmap_information_header));
      }
   };

   template <typename T>
   inline void write_to_stream(std::ofstream& stream,const T& t)
   {
      stream.write(reinterpret_cast<const char*>(&t),sizeof(T));
   }

   inline unsigned short flip(const unsigned short& v)
   {
      return ((v >> 8) | (v << 8));
   }

   inline unsigned int flip(const unsigned int& v)
   {
      return (
               ((v & 0xFF000000) >> 0x18) |
               ((v & 0x000000FF) << 0x18) |
               ((v & 0x00FF0000) >> 0x08) |
               ((v & 0x0000FF00) << 0x08)
             );
   }

      inline bool big_endian()
   {
      unsigned int v = 0x01;

      return (1 != reinterpret_cast<char*>(&v)[0]);
   }

     inline void write_bfh(std::ofstream& stream, const bitmap_file_header& bfh)
   {
      if (big_endian())
      {
         write_to_stream(stream,flip(bfh.type     ));
         write_to_stream(stream,flip(bfh.size     ));
         write_to_stream(stream,flip(bfh.reserved1));
         write_to_stream(stream,flip(bfh.reserved2));
         write_to_stream(stream,flip(bfh.off_bits ));
      }
      else
      {
         write_to_stream(stream,bfh.type     );
         write_to_stream(stream,bfh.size     );
         write_to_stream(stream,bfh.reserved1);
         write_to_stream(stream,bfh.reserved2);
         write_to_stream(stream,bfh.off_bits );
      }
   }

      inline void write_bih(std::ofstream& stream, const bitmap_information_header& bih)
   {
      if (big_endian())
      {
         write_to_stream(stream,flip(bih.size            ));
         write_to_stream(stream,flip(bih.width           ));
         write_to_stream(stream,flip(bih.height          ));
         write_to_stream(stream,flip(bih.planes          ));
         write_to_stream(stream,flip(bih.bit_count       ));
         write_to_stream(stream,flip(bih.compression     ));
         write_to_stream(stream,flip(bih.size_image      ));
         write_to_stream(stream,flip(bih.x_pels_per_meter));
         write_to_stream(stream,flip(bih.y_pels_per_meter));
         write_to_stream(stream,flip(bih.clr_used        ));
         write_to_stream(stream,flip(bih.clr_important   ));
      }
      else
      {
         write_to_stream(stream,bih.size            );
         write_to_stream(stream,bih.width           );
         write_to_stream(stream,bih.height          );
         write_to_stream(stream,bih.planes          );
         write_to_stream(stream,bih.bit_count       );
         write_to_stream(stream,bih.compression     );
         write_to_stream(stream,bih.size_image      );
         write_to_stream(stream,bih.x_pels_per_meter);
         write_to_stream(stream,bih.y_pels_per_meter);
         write_to_stream(stream,bih.clr_used        );
         write_to_stream(stream,bih.clr_important   );
      }
   }

void save_image(char * filename, int width, int height, unsigned char * img_ptr){
   std::ofstream stream(filename,std::ios::binary);

      bitmap_information_header bih;

      bih.width            = width;
      bih.height           = height;
      bih.bit_count        = static_cast<unsigned short>(3 << 3);
      bih.clr_important    = 0;
      bih.clr_used         = 0;
      bih.compression      = 0;
      bih.planes           = 1;
      bih.size             = bih.struct_size();
      bih.x_pels_per_meter = 0;
      bih.y_pels_per_meter = 0;
      bih.size_image       = (((bih.width * 3) + 3) & 0x0000FFFC) * bih.height;

      bitmap_file_header bfh;

      bfh.type             = 19778;
      bfh.size             = bfh.struct_size() + bih.struct_size() + bih.size_image;
      bfh.reserved1        = 0;
      bfh.reserved2        = 0;
      bfh.off_bits         = bih.struct_size() + bfh.struct_size();

      write_bfh(stream,bfh);
      write_bih(stream,bih);

      unsigned int padding = (4 - ((3 * width) % 4)) % 4;
      char padding_data[4] = { 0x00, 0x00, 0x00, 0x00 };

      for (unsigned int i = 0; i < height; ++i)
      {
         unsigned int row_increment_ = width * 3;
         const unsigned char* data_ptr = &img_ptr[(row_increment_ * (height - i - 1))];

         stream.write(reinterpret_cast<const char*>(data_ptr), sizeof(unsigned char) * 3 * width);
         stream.write(padding_data,padding);
      }

      stream.close();
}